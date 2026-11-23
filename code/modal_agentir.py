"""Modal entrypoint. Default action is a bounded benchmark, never full training.

    modal run --detach modal_agentir.py --mode benchmark
    modal run --detach modal_agentir.py --mode train --run-id <ID> --seconds 25200
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from agentir_config import (BENCHMARK_SECONDS, BUILD_PACKAGES, CPU_CORES, GPU, MEMORY_MIB,
                           PACKAGES, TEVATRON_COMMIT, atomic_json,
                           estimated_hourly_rate, validate_run)

HERE = Path(__file__).resolve().parent
ROOT = "/data/agentir"
VOLUME_NAME = "ir646-agentir"
app = modal.App("ir646-agentir-06b")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "libsndfile1")
    .env({"DS_BUILD_OPS": "0", "MAX_JOBS": "2", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": ROOT + "/cache/huggingface", "OMP_NUM_THREADS": "4",
          "PYTHONPATH": "/root", "TORCH_CUDA_ARCH_LIST": "8.0"})
    .pip_install(*PACKAGES[:2])
    .pip_install(*PACKAGES[2:])
    .pip_install(*BUILD_PACKAGES)
    .run_commands("python -c 'import torch, wheel, setuptools, packaging, psutil, ninja'")
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .pip_install(f"git+https://github.com/texttron/tevatron.git@{TEVATRON_COMMIT}", extra_options="--no-deps")
)
for name in ("agentir_config.py", "agentir_data.py", "agentir_training.py", "agentir_workload.py",
             "modal_agentir.py", "agentir_deepspeed.json"):
    image = image.add_local_file(str(HERE / name), f"/root/{name}")

resources = dict(image=image, volumes={"/data": volume}, cpu=(CPU_CORES, CPU_CORES),
                 memory=(MEMORY_MIB, MEMORY_MIB), retries=0, max_containers=1,
                 scaledown_window=2)


@app.function(**resources, timeout=1800)
def prepare_data(run_id: str = "", full_replay: bool = False, corpus: bool = False,
                 precision: str = "fp16"):
    from agentir_data import prepare, prepare_full_replay
    from agentir_config import CORPUS_ID
    from huggingface_hub import hf_hub_download
    if not run_id:
        result = prepare(ROOT, volume.commit, precision=precision)
        run_id = result["run_id"]
    else:
        result = {"run_id": run_id}
    run = Path(ROOT) / "experiments" / run_id
    if not (run / "manifest.json").exists():
        raise ValueError("Unknown run ID; run the benchmark preparation first")
    validate_run(run)
    if full_replay:
        result["replay"] = prepare_full_replay(ROOT, run_id, volume.commit)
    if corpus:
        manifest = json.loads((run / "manifest.json").read_text())
        for name in json.loads((run / "corpus_files.json").read_text()):
            hf_hub_download(CORPUS_ID, name, repo_type="dataset",
                           revision=manifest["revisions"][CORPUS_ID],
                           cache_dir=ROOT + "/cache/huggingface")
        volume.commit()
    return result


@app.function(**resources, gpu=GPU, timeout=BENCHMARK_SECONDS)
def benchmark(run_id: str):
    from agentir_workload import run_benchmark
    validate_run(Path(ROOT) / "experiments" / run_id)
    return run_benchmark(ROOT, run_id, volume.commit, seconds=BENCHMARK_SECONDS)


@app.function(**resources, gpu=GPU, timeout=8 * 60 * 60)
def execute(mode: str, run_id: str, seconds: int):
    from agentir_workload import launch_training, run_encode, run_replay
    started = time.perf_counter()
    if mode == "precision-trial":
        if not 120 <= seconds <= 600:
            raise ValueError("Precision trial duration must be between 120 and 600 seconds")
    elif mode not in {"train", "encode", "replay"} or not 120 <= seconds <= 7 * 60 * 60:
        raise ValueError("Use train/encode/replay with a duration between 120 and 25200 seconds")
    run = Path(ROOT) / "experiments" / run_id
    manifest = validate_run(run)
    if mode == "precision-trial":
        config = manifest["config"]
        valid_bf16 = config["bf16"] and config["attention"] != "sdpa"
        valid_fp32 = not config["bf16"] and config["attention"] == "sdpa"
        if config["fp16"] or not (valid_bf16 or valid_fp32):
            raise ValueError("Precision trial must use BF16/FlashAttention or FP32/SDPA")
        warmstart_path = run / "training" / "warmstart_adapter.pt"
        expected_precision = "bf16" if config["bf16"] else "fp32"
        resume_origin_path = run / "training" / "resume_origin.json"
        if resume_origin_path.is_file():
            resume_origin = json.loads(resume_origin_path.read_text())
            if resume_origin.get("precision") != "bf16" or not resume_origin.get("optimizer_state_restored"):
                raise ValueError("BF16 checkpoint provenance is invalid")
            if not any((run / "training").glob("checkpoint-*/checkpoint_complete.json")):
                raise ValueError("Complete BF16 resume checkpoint is missing")
        else:
            origin_path = run / "training" / "warmstart_origin.json"
            if not warmstart_path.is_file() or not origin_path.is_file():
                raise ValueError("Precision adapter warm start and provenance are required")
            if json.loads(origin_path.read_text()).get("precision") != expected_precision:
                raise ValueError("Warm-start precision does not match the run manifest")
    else:
        # One GPU worker at a time; benchmark and full stages should be launched sequentially.
        latest = json.loads((run / "benchmark_latest.json").read_text())
        measured = json.loads(Path(latest["path"]).read_text())
        if measured["status"] != "complete":
            raise ValueError("Obtain a complete benchmark before launching a full stage")
        projected = measured["projection"]["with_25_percent_runtime_margin_usd"]
        if projected > 40:
            raise ValueError(f"Projected compute ${projected:.2f} exceeds the $40 project budget; revise the plan first")
    # This is an estimate for this project, not a replacement for Modal's Workspace budget.
    receipts = list((run / "usage").glob("*.json"))
    reserved = sum(json.loads(p.read_text())["reserved_compute_usd"] for p in receipts)
    estimate = (seconds + 60) / 3600 * estimated_hourly_rate()
    if reserved + estimate > 38:
        raise ValueError("Insufficient estimated project budget; reserve $2 for preparation/build/storage overhead")
    receipt = run / "usage" / (str(time.time_ns()) + ".json")
    atomic_json(receipt, {"mode": mode, "reserved_compute_usd": estimate, "status": "reserved"})
    volume.commit()
    try:
        if mode in {"train", "precision-trial"}:
            if (run / "training" / "training_complete.json").exists():
                result = {"status": "complete", "already_completed": True}
            else:
                result = launch_training(run, run / "training", seconds - 30,
                                         max_new_steps=25 if mode == "precision-trial" else 0)
        elif mode == "encode":
            result = run_encode(ROOT, run_id, started + seconds, volume.commit)
        else:
            result = run_replay(ROOT, run_id, started + seconds, volume.commit)
        return result
    finally:
        elapsed = time.perf_counter() - started
        atomic_json(receipt, {"mode": mode, "elapsed_seconds": elapsed,
                              "reserved_compute_usd": (elapsed + 10) / 3600 * estimated_hourly_rate(),
                              "status": "finished"})
        volume.commit()


@app.function(image=image, volumes={"/data": volume}, cpu=(0.125, 0.125),
              memory=(512, 1024), timeout=9 * 60 * 60, retries=0, max_containers=1,
              scaledown_window=2)
def pipeline(mode: str = "benchmark", run_id: str = "", seconds: int = 1800,
             precision: str = "fp16"):
    """Keep preparation and subsequent GPU dispatch on Modal after disconnection."""
    if mode not in {"benchmark", "prepare", "train", "encode", "replay", "precision-trial"}:
        raise ValueError("mode must be benchmark, prepare, train, encode, replay, or precision-trial")
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("precision must be fp16, bf16, or fp32")
    if mode in {"train", "encode", "replay", "precision-trial"} and not run_id:
        raise ValueError("Supply the prepared run ID; stages never launch implicitly")
    if mode in {"benchmark", "prepare"}:
        prepared = prepare_data.remote(run_id, precision=precision)
        run_id = prepared["run_id"]
        result = benchmark.remote(run_id) if mode == "benchmark" else prepared
    else:
        if mode == "encode":
            prepare_data.remote(run_id, corpus=True)
        if mode == "replay":
            prepare_data.remote(run_id, full_replay=True)
        result = execute.remote(mode, run_id, seconds)
    result = {**result, "run_id": run_id}
    # Child workers committed their writes; refresh this coordinator's Volume view.
    volume.reload()
    remote_report = Path(ROOT) / "experiments" / run_id / "reports" / f"{mode}_{time.time_ns()}.json"
    atomic_json(remote_report, result)
    volume.commit()
    print(json.dumps({"run_id": run_id, "status": result.get("status", "prepared"),
                      "volume_report": str(remote_report), "projection": result.get("projection")}, indent=2))
    return result


@app.local_entrypoint()
def main(mode: str = "benchmark", run_id: str = "", seconds: int = 1800,
         precision: str = "fp16"):
    # One cloud call owns the whole sequence; the local process only mirrors results.
    result = pipeline.remote(mode, run_id, seconds, precision)
    run_id = result["run_id"]
    target = HERE / "modal_results" / f"{run_id}_{mode}_{time.time_ns()}.json"
    atomic_json(target, result)
    print(json.dumps({"run_id": run_id, "status": result.get("status", "prepared"),
                      "local_report": str(target), "remote_root": f"{ROOT}/experiments/{run_id}",
                      "projection": result.get("projection")}, indent=2))
