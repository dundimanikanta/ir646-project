"""Pinned Tevatron/DeepSpeed training worker, launched with torchrun on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from agentir_config import TrainingConfig, atomic_json, training_steps


def configure_loss_scaler(optimizer, configured_minimum):
    scaler = getattr(optimizer, "loss_scaler", None)
    if scaler is None:
        raise RuntimeError("FP16 training expected a DeepSpeed dynamic loss scaler")
    minimum = float(configured_minimum)
    if minimum == 0:
        minimum = 1 / 1024
    scaler.min_scale = minimum
    return {
        "current": float(scaler.cur_scale),
        "minimum": float(scaler.min_scale),
        "raise_error_at_minimum": bool(scaler.raise_error_at_min_scale),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-new-steps", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    deadline = started + args.seconds
    run, output = Path(args.run_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "timing.json"
    report = {"status": "initializing", "benchmark": args.benchmark,
              "steps": [], "synthetic_stress_microbatches": 0}
    atomic_json(report_path, report)

    import torch
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
    from tevatron.retriever.collator import TrainCollator
    from tevatron.retriever.dataset import TrainDataset
    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.trainer import TevatronTrainer

    manifest = json.loads((run / "manifest.json").read_text())
    config = TrainingConfig(**manifest["config"])
    model_path = json.loads((run / "prepared.json").read_text())["model_path"]
    set_seed(config.seed)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # torchrun supplies rank information. Initializing here avoids an MPI dependency.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_args = TevatronTrainingArguments(
        output_dir=str(output), overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        num_train_epochs=config.epochs, learning_rate=config.learning_rate,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        fp16=config.fp16, bf16=config.bf16, gradient_checkpointing=True,
        max_grad_norm=config.max_grad_norm, seed=config.seed, data_seed=config.seed,
        deepspeed=manifest["deepspeed"],
        save_strategy="no" if args.benchmark else "steps",
        save_steps=config.save_steps, save_total_limit=2,
        logging_strategy="steps", logging_steps=1, logging_nan_inf_filter=False,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_drop_last=False, disable_tqdm=False,
    )
    # The full epoch schedule is retained even during the time-limited benchmark.
    model_args = ModelArguments(
        model_name_or_path=model_path, pooling="last", normalize=True,
        temperature=config.temperature, lora=True, lora_r=config.lora_r,
        lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        attn_implementation=config.attention,
    )
    from agentir_config import QUERY_PREFIX
    data_args = DataArguments(
        dataset_name="json", dataset_path=str(run / "train_query_preserved.jsonl"),
        dataset_cache_dir=str(run / "dataset_cache"), train_group_size=config.train_group_size,
        query_prefix=QUERY_PREFIX, passage_prefix="", query_max_len=config.query_tokens,
        passage_max_len=config.passage_tokens, padding_side="left", pad_to_multiple_of=16,
    )
    model_dtype = torch.float16 if config.fp16 else torch.bfloat16 if config.bf16 else torch.float32
    model = DenseModel.build(model_args, train_args, torch_dtype=model_dtype,
                             attn_implementation=config.attention)
    model.encoder.config.use_cache = False
    # Compatibility attribute expected by HF resume warnings on an nn.Module wrapper.
    model._keys_to_ignore_on_save = None
    complete_checkpoints = [
        path for path in output.glob("checkpoint-*")
        if (path / "checkpoint_complete.json").exists()
    ]
    warmstart_path = output / "warmstart_adapter.pt"
    if warmstart_path.is_file() and not complete_checkpoints:
        adapter_state = torch.load(warmstart_path, map_location="cpu", weights_only=True)
        if not adapter_state or any("lora_" not in key for key in adapter_state):
            raise ValueError("Warm-start file must contain only non-empty LoRA adapter weights")
        loaded = model.load_state_dict(adapter_state, strict=False)
        if loaded.unexpected_keys:
            raise ValueError(f"Unexpected warm-start adapter keys: {loaded.unexpected_keys[:5]}")
        missing_adapter_keys = [key for key in loaded.missing_keys if "lora_" in key]
        if missing_adapter_keys:
            raise ValueError(f"Warm-start adapter weights were not loaded: {missing_adapter_keys[:5]}")
        report["warmstart_adapter_loaded"] = True
        report["optimizer_state_reset"] = True
        origin_path = output / "warmstart_origin.json"
        if origin_path.is_file():
            report["warmstart_origin"] = json.loads(origin_path.read_text())
    report["precision"] = "fp16" if config.fp16 else "bf16" if config.bf16 else "fp32"
    dataset = TrainDataset(data_args)
    official_collator = TrainCollator(data_args, tokenizer)

    class ObservedCollator:
        calls = 0

        def __call__(self, features):
            query, passage = official_collator(features)
            # One complete accumulation window stresses the actual maximum shapes.
            # Synthetic inputs never enter full training or retrieval-quality reports.
            if args.benchmark and self.calls < config.gradient_accumulation:
                for tokens, limit in [(query, config.query_tokens), (passage, config.passage_tokens)]:
                    stretched = []
                    for ids, mask in zip(tokens["input_ids"], tokens["attention_mask"]):
                        valid = ids[mask.bool()]
                        repeats = math.ceil(limit / max(len(valid), 1))
                        stretched.append(valid.repeat(repeats)[:limit])
                    tokens["input_ids"] = torch.stack(stretched)
                    tokens["attention_mask"] = torch.ones_like(tokens["input_ids"])
                report["synthetic_stress_microbatches"] += 1
            self.calls += 1
            report["latest_shapes"] = {"queries": list(query["input_ids"].shape),
                                       "passages": list(passage["input_ids"].shape)}
            return query, passage

    class TimingCallback(TrainerCallback):
        step_started = None
        initial_step = 0
        previous_skips = 0
        trainer = None

        def on_train_begin(self, arguments, state, control, **kwargs):
            self.initial_step = state.global_step
            report["initial_global_step"] = state.global_step
            report["setup_seconds"] = time.perf_counter() - started
            report["status"] = "training"
            if config.fp16:
                optimizer = getattr(self.trainer.model_wrapped, "optimizer", None)
                report["loss_scaler_at_start"] = configure_loss_scaler(
                    optimizer, manifest["deepspeed"]["fp16"]["min_loss_scale"])
            else:
                report["loss_scaler_at_start"] = None
            self.previous_skips = int(getattr(self.trainer.model_wrapped, "skipped_steps", 0))
            torch.cuda.reset_peak_memory_stats()

        def on_step_begin(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.step_started
            engine = self.trainer.model_wrapped
            skips = getattr(engine, "skipped_steps", None)
            skipped = None if skips is None else int(skips) > self.previous_skips
            if skips is not None:
                self.previous_skips = int(skips)
            optimizer = getattr(engine, "optimizer", None)
            scaler = getattr(optimizer, "loss_scaler", None)
            scale = getattr(scaler, "cur_scale", None)
            report["steps"].append({"global_step": state.global_step, "seconds": elapsed,
                                    "skipped": skipped,
                                    "loss_scale": float(scale) if config.fp16 and scale is not None else None})
            report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            report["elapsed_seconds"] = time.perf_counter() - started
            report["gradient_checkpointing_enabled"] = bool(model.encoder.is_gradient_checkpointing)
            atomic_json(report_path, report)
            enough_steps = args.max_new_steps and state.global_step - self.initial_step >= args.max_new_steps
            if time.perf_counter() >= deadline - 30 or enough_steps:
                control.should_training_stop = True
                control.should_save = not args.benchmark
            return control

        def on_log(self, arguments, state, control, logs=None, **kwargs):
            report["last_training_log"] = {
                k: (float(v) if math.isfinite(v) else str(v))
                for k, v in (logs or {}).items() if isinstance(v, (int, float))
            }
            atomic_json(report_path, report)

        def on_save(self, arguments, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            # Completion marker makes a partially written checkpoint ineligible for resume.
            atomic_json(checkpoint / "checkpoint_complete.json", {"global_step": state.global_step})
            report["last_checkpoint"] = str(checkpoint)
            atomic_json(report_path, report)
            try:
                import modal
                modal.Volume.from_name("ir646-agentir").commit()
            except ImportError:
                pass

    observer = TimingCallback()
    trainer = TevatronTrainer(model=model, args=train_args, train_dataset=dataset,
                              data_collator=ObservedCollator(), tokenizer=tokenizer,
                              callbacks=[observer])
    observer.trainer = trainer
    dataset.set_trainer(trainer)
    checkpoints = sorted((p for p in output.glob("checkpoint-*")
                          if (p / "checkpoint_complete.json").exists()),
                         key=lambda p: int(p.name.split("-")[-1]))
    if get_last_checkpoint(str(output)) and not checkpoints:
        raise RuntimeError("Only incomplete checkpoints exist; inspect them before starting a new run")
    resume = str(checkpoints[-1]) if checkpoints and not args.benchmark else None
    try:
        trainer.train(resume_from_checkpoint=resume)
        completed = not args.benchmark and trainer.state.global_step >= training_steps(config=config)
        trial_complete = (
            not args.benchmark and args.max_new_steps > 0
            and trainer.state.global_step - report.get("initial_global_step", 0) >= args.max_new_steps
        )
        adapter = output / ("final_adapter" if completed else "benchmark_adapter" if args.benchmark else "partial_adapter")
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        report.update(status="complete" if completed else "benchmark_complete" if args.benchmark
                      else "precision_trial_complete" if trial_complete else "paused_at_time_limit",
                      global_step=trainer.state.global_step, adapter_path=str(adapter),
                      elapsed_seconds=time.perf_counter() - started)
        if completed:
            atomic_json(output / "training_complete.json", {"adapter_path": str(adapter),
                        "global_step": trainer.state.global_step, "manifest": str(run / "manifest.json")})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        atomic_json(report_path, report)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
