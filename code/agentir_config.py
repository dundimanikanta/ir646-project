"""Experiment settings and cost arithmetic; safe to import without ML packages."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

AGENTIR_COMMIT = "9102796661c0cbc4e6c8c0a9d9bcaca667a8345b"
# The Tevatron commit in the pinned AgentIR uv.lock, rather than today's main.
TEVATRON_COMMIT = "6555f12c844c64f020f5762096405a7ce5148b77"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISIONS = {
    MODEL_ID: "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "Tevatron/AgentIR-data": "49de17cf11ba1ea4c5ae7699e2d146ca5d4ab6f2",
    "Tevatron/browsecomp-plus": "144cff8e35b5eaef7e526346aa60774a9deb941f",
    "liuqi6777/bcp_search_agent_trajectory": "af778a3994cfcb30a8a9ab5abcbcb7c7fcbf645f",
}
BCP_COMMIT = "046949032b0328319cc9a02663a759ec601d9402"
CORPUS_ID = "Tevatron/browsecomp-plus-corpus"
TRAJECTORY_ID = "liuqi6777/bcp_search_agent_trajectory"
AGENTS = ("tongyi-dr", "gpt-oss-120b-high", "qwen35-35b-a3b", "ds4pro", "kimi26", "glm51")
QUERY_PREFIX = (
    "Instruct: Given a user's reasoning followed by a web search query, "
    "retrieve relevant passages that answer the query while incorporating "
    "the user's reasoning\nQuery:"
)
EXPECTED_TRAINING_ROWS = 5238
EXPECTED_DOCUMENTS = 100195
EXPECTED_QUESTIONS = 830
EXPECTED_SEARCHES = 127600
GPU = "A100-80GB"
CPU_CORES = 4
MEMORY_MIB = 32768
BENCHMARK_SECONDS = 600
# Standard Function list rates checked 2026-10-30, NOT a billing/account cap.
GPU_USD_SECOND = 0.000694
CPU_USD_CORE_SECOND = 0.0000131
MEMORY_USD_GIB_SECOND = 0.00000222
RATE_SOURCE = "https://modal.com/pricing"

# FlashAttention is installed with --no-build-isolation, so its build tools
# must already exist in the image. Keep this as a separate layer so the large
# Torch/ML dependency layers remain reusable after a build-tool fix.
BUILD_PACKAGES = ("wheel==0.45.1", "setuptools==80.9.0")

PACKAGES = (
    "torch==2.7.1", "torchvision==0.22.1",
    "transformers==4.53.2", "peft==0.18.0", "accelerate==1.10.1",
    "datasets==4.0.0", "huggingface_hub==0.36.0", "deepspeed==0.17.2",
    "numpy==1.26.4", "pandas==2.2.3", "pyarrow==20.0.0",
    "faiss-cpu==1.11.0.post1", "qwen-omni-utils==0.0.8",
    "pillow==11.2.1", "requests==2.32.4", "ninja==1.11.1.4", "packaging==25.0",
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    temperature: float = 0.01
    train_group_size: int = 8
    query_tokens: int = 8192
    passage_tokens: int = 4096
    eval_query_tokens: int = 8192
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    attention: str = "flash_attention_2"
    doc_batch_size: int = 4
    query_batch_size: int = 4
    search_batch_size: int = 64
    top_k: int = 10
    save_steps: int = 25
    input_variant: str = "raw_current_reasoning_plus_query"
    # Explicit adaptation: keep the current query instead of right-clipping it.
    query_truncation: str = "trim_reasoning_start_preserve_current_query_v2"
    passage_truncation: str = "right"

    def __post_init__(self):
        for name in ("epochs", "batch_size", "gradient_accumulation", "train_group_size",
                     "query_tokens", "passage_tokens", "eval_query_tokens", "save_steps",
                     "doc_batch_size", "query_batch_size", "search_batch_size", "top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if max(self.query_tokens, self.passage_tokens, self.eval_query_tokens) > 32768:
            raise ValueError("Qwen3-Embedding-0.6B context limit is 32768 tokens")
        if self.eval_query_tokens != self.query_tokens:
            raise ValueError("Keep train/replay query limits equal for this baseline")
        if self.fp16 and self.bf16:
            raise ValueError("FP16 and BF16 cannot both be enabled")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def training_steps(n=EXPECTED_TRAINING_ROWS, config=None) -> int:
    c = config or TrainingConfig()
    return c.epochs * math.ceil(math.ceil(n / c.batch_size) / c.gradient_accumulation)


def estimated_hourly_rate() -> float:
    return 3600 * (GPU_USD_SECOND + CPU_CORES * CPU_USD_CORE_SECOND
                   + MEMORY_MIB / 1024 * MEMORY_USD_GIB_SECOND)


def cost_projection(training_seconds_per_step, document_seconds_per_item,
                    query_seconds_per_item, search_seconds_per_item,
                    setup_seconds=0, overhead_fraction=0.25):
    rates = (training_seconds_per_step, document_seconds_per_item,
             query_seconds_per_item, search_seconds_per_item, setup_seconds)
    if any(not math.isfinite(x) or x < 0 for x in rates):
        raise ValueError("Timings must be finite and non-negative")
    seconds = {
        "training": training_steps() * training_seconds_per_step,
        "corpus_encoding": EXPECTED_DOCUMENTS * document_seconds_per_item,
        "query_encoding": EXPECTED_SEARCHES * query_seconds_per_item,
        "exact_search": EXPECTED_SEARCHES * search_seconds_per_item,
        "measured_setup": setup_seconds,
    }
    base = sum(seconds.values())
    return {"seconds_by_stage": seconds, "estimated_hours": base / 3600,
            "estimated_compute_usd": base / 3600 * estimated_hourly_rate(),
            "with_25_percent_runtime_margin_usd": base / 3600 * estimated_hourly_rate() * (1 + overhead_fraction),
            "hourly_rate_usd": estimated_hourly_rate(),
            "excluded": ["image build", "unmeasured downloads/tokenization/checkpoint I/O", "storage", "taxes", "other account jobs"],
            "not_a_billing_guarantee": True}


def make_manifest(config: TrainingConfig, revisions: dict, source_hashes: dict,
                  deepspeed_config: dict) -> dict:
    if CORPUS_ID not in revisions or not revisions[CORPUS_ID]:
        raise ValueError("Resolve and persist the corpus revision before fingerprinting")
    return {"implementation": "pinned_tevatron_deepspeed_v1", "model": MODEL_ID,
            "config": asdict(config), "revisions": revisions,
            "agentir_commit": AGENTIR_COMMIT, "tevatron_commit": TEVATRON_COMMIT,
            "deepspeed": deepspeed_config, "local_source_sha256": source_hashes,
            "image": "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04;python=3.11",
            "packages": list(PACKAGES) + list(BUILD_PACKAGES) + ["flash-attn==2.8.3"],
            "evaluation": {"agents": list(AGENTS), "questions": EXPECTED_QUESTIONS,
                           "documents": EXPECTED_DOCUMENTS, "qrels_commit": BCP_COMMIT,
                           "protocol": "fixed_offline_replay_all_830_no_model_selection"}}


def validate_run(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if fingerprint(manifest) != run.name:
        raise ValueError("Run directory and manifest fingerprint differ")
    here = Path(__file__).parent
    for name, expected in manifest["local_source_sha256"].items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} changed since this run was prepared; create a new run instead of reusing checkpoints")
    return manifest
