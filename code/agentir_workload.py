"""Bounded GPU workloads. Benchmark artifacts never count as trained baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from agentir_config import (AGENTS, EXPECTED_DOCUMENTS, EXPECTED_SEARCHES,
                           QUERY_PREFIX, TrainingConfig, atomic_json,
                           cost_projection, estimated_hourly_rate, fingerprint)
from agentir_data import fit_query


def launch_training(run, output, seconds, benchmark=False, max_new_steps=0):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
               str(Path(__file__).with_name("agentir_training.py")),
               "--run-dir", str(run), "--output-dir", str(output), "--seconds", str(seconds)]
    if benchmark:
        command += ["--benchmark", "--max-new-steps", "30"]
    elif max_new_steps:
        command += ["--max-new-steps", str(max_new_steps)]
    output.mkdir(parents=True, exist_ok=True)
    # A child process releases DeepSpeed's GPU allocations before embedding begins.
    with (output / "console.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_deadline = time.perf_counter() + seconds + 60
        last_step = None
        try:
            while True:
                remaining = wait_deadline - time.perf_counter()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds + 5)
                try:
                    status = process.wait(timeout=min(15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    timing_path = output / "timing.json"
                    if timing_path.exists():
                        current = json.loads(timing_path.read_text())
                        steps = current.get("steps", [])
                        step = steps[-1]["global_step"] if steps else 0
                        if step != last_step:
                            print(f"Training status={current['status']}, optimizer-step attempts={step}", flush=True)
                            last_step = step
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status = -1
    timing = output / "timing.json"
    result = json.loads(timing.read_text()) if timing.exists() else {"status": "failed"}
    result["process_exit_code"] = status
    if status != 0:
        was_watchdog_timeout = status == -1 and result.get("steps") and not result.get("error")
        result["status"] = "paused_at_time_limit" if was_watchdog_timeout else "failed_or_timed_out"
        result["log_path"] = str(output / "console.log")
    return result


def load_encoder(run, adapter):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    prepared = json.loads((run / "prepared.json").read_text())
    config = TrainingConfig(**json.loads((run / "manifest.json").read_text())["config"])
    tokenizer = AutoTokenizer.from_pretrained(prepared["model_path"], padding_side="left")
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(prepared["model_path"], torch_dtype=torch.float16,
                                      attn_implementation=config.attention)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().cuda().eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    return model, tokenizer, config


def encode(model, tokenizer, texts, limit, batch_size, is_query=False):
    import numpy as np
    import torch
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=not is_query,
                               max_length=limit if not is_query else None,
                               pad_to_multiple_of=16, return_tensors="pt")
            if tokens["input_ids"].shape[1] > limit:
                raise ValueError("Formatted query exceeds its cap; refusing to truncate the current query")
            tokens = {k: v.cuda() for k, v in tokens.items()}
            vectors = torch.nn.functional.normalize(model(**tokens).last_hidden_state[:, -1].float(), dim=-1)
            if not torch.isfinite(vectors).all():
                raise ValueError("Non-finite embeddings")
            result.append(vectors.cpu().numpy())
    return np.ascontiguousarray(np.concatenate(result), dtype=np.float32)


def query_text(step, tokenizer, limit):
    raw = "Reasoning: " + step["reasoning"] + "\nQuery: " + step["query"]
    fitted, clipped = fit_query(raw, tokenizer, limit)
    return QUERY_PREFIX + fitted, clipped


def run_benchmark(root, run_id, commit=lambda: None, seconds=600):
    import numpy as np
    import torch
    import faiss

    started = time.perf_counter()
    deadline = started + min(seconds, 600) - 25
    run = Path(root) / "experiments" / run_id
    bench_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = run / "benchmarks" / bench_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "benchmark.json"
    report = {"run_id": run_id, "benchmark_id": bench_id, "status": "running",
              "gpu": torch.cuda.get_device_name(), "budget_seconds": min(seconds, 600),
              "sample_warning": json.loads((run / "prepared.json").read_text())["sample_warning"],
              "search_timing_uses_synthetic_vectors": True,
              "retrieval_quality_evaluated": False}
    from importlib.metadata import version
    report["runtime_versions"] = {name: version(name) for name in
                                  ["torch", "transformers", "peft", "accelerate", "deepspeed", "flash-attn"]}

    def persist():
        report["gpu_function_seconds"] = time.perf_counter() - started
        report["estimated_gpu_function_compute_usd"] = report["gpu_function_seconds"] / 3600 * estimated_hourly_rate()
        atomic_json(path, report)
        atomic_json(run / "benchmark_latest.json", {"path": str(path)})
        commit()

    persist()
    try:
        train_seconds = min(300, max(60, (deadline - time.perf_counter()) * 0.55))
        print(f"Benchmark: training phase, at most {train_seconds:.0f} seconds", flush=True)
        report["training"] = launch_training(run, folder / "training", train_seconds, benchmark=True)
        persist()
        training = report["training"]
        if training["status"] != "benchmark_complete":
            raise RuntimeError("Training benchmark did not finish; inspect its timing.json and console.log")
        model, tokenizer, config = load_encoder(run, training["adapter_path"])
        faiss.omp_set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        pools = {
            "documents": [r["text"] for r in json.loads((run / "benchmark_documents.json").read_text())],
            "queries": [query_text(r, tokenizer, config.eval_query_tokens)[0]
                        for r in json.loads((run / "benchmark_queries.json").read_text())],
        }
        query_vectors = []
        for name, texts in pools.items():
            print(f"Benchmark: {name} encoding, {len(texts)} sampled inputs", flush=True)
            limit = config.passage_tokens if name == "documents" else config.eval_query_tokens
            batch_size = config.doc_batch_size if name == "documents" else config.query_batch_size
            durations, counts = [], []
            for start in range(0, len(texts), batch_size):
                if time.perf_counter() > deadline - 40:
                    break
                chunk = texts[start:start + batch_size]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                vectors = encode(model, tokenizer, chunk, limit, batch_size, is_query=name == "queries")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                # First batch includes cold inference setup; exclude it from throughput.
                if start:
                    durations.append(elapsed)
                    counts.append(len(chunk))
                if name == "queries":
                    query_vectors.append(vectors)
            report[name] = {"measured_items": sum(counts), "measured_seconds": sum(durations),
                            "seconds_per_item": sum(durations) / sum(counts) if counts else None,
                            "batches": len(durations)}
            persist()
        if not query_vectors or time.perf_counter() > deadline - 20:
            raise RuntimeError("Insufficient time to measure exact search; rerun with the warmed cache")
        # Full-size FAISS timing without paying to encode the entire corpus first.
        index = faiss.IndexFlatIP(1024)
        rng = np.random.default_rng(config.seed)
        for start in range(0, EXPECTED_DOCUMENTS, 4096):
            vectors = rng.standard_normal((min(4096, EXPECTED_DOCUMENTS - start), 1024), dtype=np.float32)
            faiss.normalize_L2(vectors)
            index.add(vectors)
        vectors = np.concatenate(query_vectors)
        index.search(vectors[:min(8, len(vectors))], config.top_k)
        begin = time.perf_counter()
        for start in range(0, len(vectors), config.search_batch_size):
            index.search(vectors[start:start + config.search_batch_size], config.top_k)
        search_seconds = time.perf_counter() - begin
        report["search"] = {"documents": index.ntotal, "queries": len(vectors),
                            "seconds_per_item": search_seconds / len(vectors)}
        report["inference_peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        steady = training["steps"][2:]
        successful = [s for s in steady if s["skipped"] is False]
        if len(successful) < 3 or any(report[k]["seconds_per_item"] is None for k in ["documents", "queries"]):
            raise RuntimeError("Too few successful steady-state measurements for a cost projection")
        step_seconds = statistics.mean(s["seconds"] for s in successful)
        report["steady_training_steps"] = len(successful)
        report["training_seconds_per_successful_step"] = step_seconds
        report["projection"] = cost_projection(
            step_seconds, report["documents"]["seconds_per_item"], report["queries"]["seconds_per_item"],
            report["search"]["seconds_per_item"], setup_seconds=training["setup_seconds"])
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, error=str(exc)[:1000])
    finally:
        persist()
    return report


def completed_adapter(run):
    marker = run / "training" / "training_complete.json"
    if not marker.exists():
        raise RuntimeError("Full training has not completed; benchmark/partial adapters cannot produce baseline results")
    adapter = Path(json.loads(marker.read_text())["adapter_path"])
    weights = adapter / "adapter_model.safetensors"
    if not weights.exists():
        raise RuntimeError("Completed adapter weights are missing")
    return adapter, hashlib.sha256(weights.read_bytes()).hexdigest()


def run_encode(root, run_id, deadline, commit=lambda: None):
    import numpy as np
    import pyarrow.parquet as pq
    import faiss
    from huggingface_hub import hf_hub_download
    from agentir_config import CORPUS_ID

    root = Path(root)
    run = root / "experiments" / run_id
    manifest = json.loads((run / "manifest.json").read_text())
    adapter, checksum = completed_adapter(run)
    config = TrainingConfig(**manifest["config"])
    signature = fingerprint({"run_id": run_id, "adapter_sha256": checksum,
                             "corpus": manifest["revisions"][CORPUS_ID], "cap": config.passage_tokens})
    folder = run / "indexes" / signature
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        atomic_json(run / "index_latest.json", {"path": str(folder), "adapter_sha256": checksum})
        commit()
        return {"status": "complete", "index_dir": str(folder)}
    model, tokenizer, config = load_encoder(run, adapter)
    sources = json.loads((run / "corpus_files.json").read_text())
    for si, name in enumerate(sources):
        source = hf_hub_download(CORPUS_ID, name, repo_type="dataset",
                                revision=manifest["revisions"][CORPUS_ID],
                                cache_dir=str(root / "cache" / "huggingface"))
        for bi, batch in enumerate(pq.ParquetFile(source).iter_batches(batch_size=128, columns=["docid", "text"])):
            if time.perf_counter() > deadline - 30:
                commit()
                return {"status": "paused_at_time_limit", "index_dir": str(folder)}
            out = folder / f"part_{si:04d}_{bi:06d}.npz"
            rows = batch.to_pylist()
            ids = [str(r["docid"]) for r in rows]
            if out.exists():
                with np.load(out) as saved:
                    if saved["docids"].tolist() != ids or str(saved["signature"]) != signature:
                        raise ValueError("Cached document chunk does not match this run")
                continue
            vectors = encode(model, tokenizer, [r["text"] for r in rows], config.passage_tokens, config.doc_batch_size)
            temp = out.with_suffix(".tmp.npz")
            np.savez(temp, embeddings=vectors.astype(np.float16), docids=np.asarray(ids), signature=signature)
            temp.replace(out)
            if bi % 10 == 0:
                commit()
    index, docids = faiss.IndexFlatIP(1024), []
    for path in sorted(folder.glob("part_*.npz")):
        with np.load(path) as part:
            vectors = np.ascontiguousarray(part["embeddings"], dtype=np.float32)
            if not np.isfinite(vectors).all() or str(part["signature"]) != signature:
                raise ValueError("Invalid corpus vectors")
            faiss.normalize_L2(vectors)
            index.add(vectors)
            docids.extend(part["docids"].tolist())
    if index.ntotal != EXPECTED_DOCUMENTS or len(set(docids)) != EXPECTED_DOCUMENTS:
        raise ValueError("Corpus index is incomplete or contains duplicate IDs")
    faiss.write_index(index, str(folder / "index.faiss.tmp"))
    (folder / "index.faiss.tmp").replace(folder / "index.faiss")
    atomic_json(folder / "docids.json", docids)
    atomic_json(folder / "complete.json", {"signature": signature, "adapter_sha256": checksum,
                                           "documents": len(docids)})
    atomic_json(run / "index_latest.json", {"path": str(folder), "adapter_sha256": checksum})
    commit()
    return {"status": "complete", "index_dir": str(folder)}


def recall_at(ranked, relevant, k=5):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at(ranked, relevant, k=10):
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    actual = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    return actual / ideal if ideal else 0.0


def evaluate(run, folder, steps):
    evidence = {q: set(v) for q, v in json.loads((run / "qrel_evidence.txt.json").read_text()).items()}
    gold = {q: set(v) for q, v in json.loads((run / "qrel_golds.txt.json").read_text()).items()}
    accumulated = defaultdict(lambda: {"count": 0, "evidence_recall_5": 0.0,
                                       "evidence_ndcg_10": 0.0, "gold_recall_5": 0.0, "union": set()})
    config = TrainingConfig(**json.loads((run / "manifest.json").read_text())["config"])
    for start in range(0, len(steps), config.search_batch_size):
        chunk = steps[start:start + config.search_batch_size]
        saved = json.loads((folder / f"batch_{start:08d}.json").read_text())
        if saved["step_ids"] != [s["step_id"] for s in chunk]:
            raise ValueError("Evaluation step IDs mismatch")
        for step in chunk:
            ranked = [h["docid"] for h in saved["results"][step["step_id"]]]
            a = accumulated[(step["agent"], step["qid"])]
            a["count"] += 1
            a["evidence_recall_5"] += recall_at(ranked, evidence[step["qid"]])
            a["evidence_ndcg_10"] += ndcg_at(ranked, evidence[step["qid"]])
            a["gold_recall_5"] += recall_at(ranked, gold[step["qid"]])
            a["union"].update(ranked[:5])
    metrics = ["evidence_recall_5", "evidence_ndcg_10", "gold_recall_5"]
    questions = []
    for row in json.loads((run / "coverage.json").read_text()):
        a = accumulated[(row["agent"], row["qid"])]
        if a["count"] != row["n_searches"]:
            raise ValueError("Incomplete replay cannot be evaluated as a full baseline")
        questions.append({**row, **{m: a[m] / a["count"] if a["count"] else None for m in metrics},
                          "trajectory_union_recall_5": len(a["union"] & evidence[row["qid"]]) / len(evidence[row["qid"]])})
    metrics += ["trajectory_union_recall_5"]
    summary = []
    for agent in AGENTS:
        rows = [r for r in questions if r["agent"] == agent]
        summary.append({"agent": agent, "questions": len(rows), "searches": sum(r["n_searches"] for r in rows),
                        **{m: statistics.mean(r[m] for r in rows if r[m] is not None) for m in metrics}})
    for name, rows in [("question_metrics.csv", questions), ("summary_by_agent.csv", summary)]:
        with (folder / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def run_replay(root, run_id, deadline, commit=lambda: None):
    import faiss
    run = Path(root) / "experiments" / run_id
    if not (run / "replay_complete.json").exists():
        raise RuntimeError("Prepare full replay data on CPU before allocating the GPU")
    adapter, checksum = completed_adapter(run)
    pointer = json.loads((run / "index_latest.json").read_text())
    if pointer["adapter_sha256"] != checksum:
        raise ValueError("Index was encoded with a different adapter")
    index_dir = Path(pointer["path"])
    if not (index_dir / "complete.json").exists():
        raise ValueError("Incomplete corpus index")
    faiss.omp_set_num_threads(4)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    docids = json.loads((index_dir / "docids.json").read_text())
    if index.ntotal != EXPECTED_DOCUMENTS or len(set(docids)) != EXPECTED_DOCUMENTS:
        raise ValueError("Unexpected index size")
    corpus_ids = set(docids)
    for name in ("qrel_evidence.txt.json", "qrel_golds.txt.json"):
        qrels = json.loads((run / name).read_text())
        if not set().union(*(set(ids) for ids in qrels.values())) <= corpus_ids:
            raise ValueError("Labeled evidence/gold documents are missing from the corpus index")
    with (run / "replay_steps.jsonl").open() as handle:
        steps = [json.loads(line) for line in handle]
    if len(steps) != EXPECTED_SEARCHES or len({s["step_id"] for s in steps}) != len(steps):
        raise ValueError("Replay coverage mismatch")
    replay_signature = fingerprint({"run_id": run_id, "adapter": checksum,
        "steps": json.loads((run / "replay_complete.json").read_text())["coverage_sha256"]})
    folder = run / "replays" / replay_signature
    folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer, config = load_encoder(run, adapter)
    for start in range(0, len(steps), config.search_batch_size):
        if time.perf_counter() > deadline - 30:
            commit()
            return {"status": "paused_at_time_limit", "results_dir": str(folder)}
        chunk = steps[start:start + config.search_batch_size]
        out = folder / f"batch_{start:08d}.json"
        expected = [s["step_id"] for s in chunk]
        if out.exists():
            saved = json.loads(out.read_text())
            if saved["signature"] != replay_signature or saved["step_ids"] != expected:
                raise ValueError("Cached replay batch mismatch")
            continue
        formatted = [query_text(s, tokenizer, config.eval_query_tokens) for s in chunk]
        texts = [x[0] for x in formatted]
        unique = list(dict.fromkeys(texts))
        vectors = encode(model, tokenizer, unique, config.eval_query_tokens, config.query_batch_size, is_query=True)
        scores, positions = index.search(vectors, config.top_k)
        ranked = [[{"docid": docids[int(i)], "score": float(v)} for v, i in zip(ss, pp) if i >= 0]
                  for ss, pp in zip(scores, positions)]
        lookup = dict(zip(unique, ranked))
        atomic_json(out, {"signature": replay_signature, "step_ids": expected,
                         "clipped_reasonings": sum(x[1] for x in formatted),
                         "results": {s["step_id"]: lookup[t] for s, t in zip(chunk, texts)}})
        if start % (10 * config.search_batch_size) == 0:
            commit()
    summary = evaluate(run, folder, steps)
    atomic_json(folder / "complete.json", {"signature": replay_signature, "summary": summary})
    commit()
    return {"status": "complete", "results_dir": str(folder), "summary": summary}
