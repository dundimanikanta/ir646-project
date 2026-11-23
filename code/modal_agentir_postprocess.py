"""Detached, budget-authorized corpus encoding and replay stages."""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

import modal_agentir as base
from agentir_config import atomic_json, estimated_hourly_rate, validate_run
from agentir_workload import run_encode, run_replay

PROJECT_TOTAL_CAP_USD = 44.0
OVERHEAD_RESERVE_USD = 2.0

app = modal.App("ir646-agentir-postprocess")
volume = base.volume
image = base.image.add_local_file(
    str(Path(__file__).resolve()),
    "/root/modal_agentir_postprocess.py",
)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(base.CPU_CORES, base.CPU_CORES),
    memory=(base.MEMORY_MIB, base.MEMORY_MIB),
    gpu=base.GPU,
    timeout=8 * 60 * 60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def execute_stage(mode: str, run_id: str, seconds: int = 1800):
    if mode not in {"encode", "replay"}:
        raise ValueError("Post-training stage must be encode or replay")
    if not 120 <= seconds <= 7 * 60 * 60:
        raise ValueError("Stage duration must be between 120 and 25200 seconds")

    started = time.perf_counter()
    run = Path(base.ROOT) / "experiments" / run_id
    validate_run(run)
    latest = json.loads((run / "benchmark_latest.json").read_text())
    measured = json.loads(Path(latest["path"]).read_text())
    if measured["status"] != "complete":
        raise ValueError("A complete benchmark is required before post-training stages")
    projected = measured["projection"]["with_25_percent_runtime_margin_usd"]
    if projected > 40:
        raise ValueError(f"Benchmark projection ${projected:.2f} exceeds the $40 projection gate")
    if not (run / "training" / "training_complete.json").is_file():
        raise RuntimeError("Completed training is required before encoding or replay")

    receipts = list((run / "usage").glob("*.json"))
    reserved = sum(json.loads(path.read_text())["reserved_compute_usd"] for path in receipts)
    estimate = (seconds + 60) / 3600 * estimated_hourly_rate()
    compute_cap = PROJECT_TOTAL_CAP_USD - OVERHEAD_RESERVE_USD
    if reserved + estimate > compute_cap:
        raise ValueError(
            f"Insufficient estimated project budget: ${reserved + estimate:.2f} would "
            f"exceed the ${compute_cap:.2f} compute cap, retaining "
            f"${OVERHEAD_RESERVE_USD:.2f} overhead reserve"
        )

    receipt = run / "usage" / f"{time.time_ns()}.json"
    atomic_json(receipt, {
        "mode": mode,
        "reserved_compute_usd": estimate,
        "status": "reserved",
    })
    volume.commit()
    try:
        deadline = started + seconds
        if mode == "encode":
            result = run_encode(base.ROOT, run_id, deadline, volume.commit)
        else:
            result = run_replay(base.ROOT, run_id, deadline, volume.commit)
        return {**result, "run_id": run_id, "mode": mode}
    finally:
        elapsed = time.perf_counter() - started
        atomic_json(receipt, {
            "mode": mode,
            "elapsed_seconds": elapsed,
            "reserved_compute_usd": (elapsed + 10) / 3600 * estimated_hourly_rate(),
            "status": "finished",
        })
        volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(0.5, 1),
    memory=(1024, 2048),
    timeout=9 * 60 * 60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def prepare_replay_data(run_id: str):
    from agentir_data import prepare_full_replay

    return prepare_full_replay(base.ROOT, run_id, volume.commit)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(0.5, 1),
    memory=(1024, 2048),
    timeout=9 * 60 * 60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def finish_postprocessing(run_id: str):
    run = Path(base.ROOT) / "experiments" / run_id
    latest = run / "index_latest.json"
    deadline = time.monotonic() + 8 * 60 * 60

    while True:
        volume.reload()
        if latest.is_file():
            index = json.loads(latest.read_text())
            if (Path(index["path"]) / "complete.json").is_file():
                break

        pending_encode = False
        for receipt in (run / "usage").glob("*.json"):
            usage = json.loads(receipt.read_text())
            if usage.get("mode") == "encode" and usage.get("status") == "reserved":
                pending_encode = True
                break
        if pending_encode:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the active corpus-encoding slice")
            time.sleep(30)
            continue

        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out before corpus encoding completed")
        result = execute_stage.remote("encode", run_id, 5400)
        if result.get("status") == "failed":
            raise RuntimeError(f"Corpus encoding failed: {result}")

    prepare_replay_data.remote(run_id)
    return execute_stage.remote("replay", run_id, 1800)
