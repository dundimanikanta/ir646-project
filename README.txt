BASELINE 4: TRAINING, REPRODUCTION, AND MODEL USE

This repository contains our DR-Synth fine-tuned Qwen3-Embedding-0.6B retriever,
the Modal training/evaluation code, the original controller notebook, and the
final BrowseComp-Plus retrieval results. The retriever embeds the CURRENT
reasoning and current search query together. The same trained encoder embeds
documents. It produces 1,024-dimensional embeddings, not generated answers.

To use the model that produced the reported scores, load the supplied adapter
as shown in USING THE TRAINED MODEL below. To train another model with the
same BF16-stage settings, follow TRAINING A FRESH MODEL ON MODAL.


HOW WE TRAINED THE MODEL

We fine-tuned Qwen/Qwen3-Embedding-0.6B using 5,238 WebShaper examples from
Tevatron/AgentIR-data, file webshaper_train.jsonl. Each example contains raw
current reasoning and a query, one positive passage, and seven hard negatives.
Other examples' passages supply in-batch negatives. Training uses the pinned
Tevatron contrastive trainer, last-token pooling, L2-normalized embeddings,
and a shared query/document encoder with LoRA adapters.

Training settings:

  Base revision             97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
  DR-Synth revision         49de17cf11ba1ea4c5ae7699e2d146ca5d4ab6f2
  LoRA                      rank 16, alpha 64, dropout 0.1
  LoRA target modules       q_proj, k_proj, v_proj, o_proj,
                            down_proj, up_proj, gate_proj
  Microbatch size           4
  Gradient accumulation     2; effective batch size 8 on one GPU
  BF16-stage schedule       2 epochs, 1,310 optimizer-step attempts
  Optimizer                 AdamW; learning rate 1e-4; weight decay 0
  AdamW parameters          betas (0.9, 0.999), epsilon 1e-8
  Learning-rate scheduler   DeepSpeed WarmupDecayLR; 10% warmup
  Contrastive temperature   0.01
  Gradient clipping         max norm 1.0
  Reasoning/query cap        8,192 tokens, including the instruction prefix
  Document cap              4,096 tokens
  Training precision        BF16, FlashAttention 2, DeepSpeed ZeRO-3
  Memory controls           gradient checkpointing
  Checkpoints               every 25 steps; retain two checkpoints
  Seed                      42
  Hardware                  one Modal A100 80GB, 4 CPU cores, 32 GiB RAM

Oversized reasoning is trimmed from its beginning while retaining the full
current query. An oversized query that cannot fit raises an error. Documents
are right-truncated. Documents have no query instruction prefix.

The actual run began in FP16. After a loss-scale failure, we loaded the LoRA
adapter from checkpoint 675 into a BF16 run with a fresh optimizer, scheduler,
and step counter. We then completed the two-epoch BF16 schedule through step
1,310. The supplied final adapter comes from that completed run. Retrieval
inference reloads and merges the adapter into an FP16 base model.

The commands below start fresh BF16 training from the pinned pretrained base.
They reproduce the BF16-stage recipe, not the earlier FP16 warm start. Exact
historical retraining would also need the step-675 adapter and the historical
recovery states, which are not part of this repository. Use the supplied final
adapter to reuse the model associated with the published result CSVs.

The complete settings and data revisions are in experiment_config.json.
The remote image uses Python 3.11, CUDA 12.6.3, PyTorch 2.7.1, Transformers
4.53.2, PEFT 0.18.0, DeepSpeed 0.17.2, and FlashAttention 2.8.3. Tevatron is
pinned to commit 6555f12c844c64f020f5762096405a7ce5148b77.


TRAINING A FRESH MODEL ON MODAL

1. Clone the repository and install the local launcher.

Run these commands in a terminal. The local machine does not need a GPU;
Modal builds the CUDA environment and runs GPU stages remotely.

```bash
git clone https://github.com/dundimanikanta/ir646-project.git
cd ir646-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-modal.txt
python -m modal setup
```

Training, benchmarking, encoding, and replay allocate paid Modal resources.
The scripts retain their original project budget gates: the benchmark projection
including its margin must be at most $40, and the main launcher's estimated
compute reservations must stay within $38, retaining $2 for other overhead.
These fixed project estimates do not set your account's billing limit. If a
long invocation's reservation is rejected but budget remains, use a shorter
--seconds value (120 to 25,200) and resume in smaller slices. This reduces the
per-invocation reservation; it does not increase the total project allowance.

2. Use an empty experiment volume and pin the released data revisions.

Use your own clean Modal workspace/environment that does not already contain
this experiment in the ir646-agentir volume. Existing checkpoints are resumed,
and completed training is reused. The run ID fingerprints configuration and
source code, not warm-start history: fresh training can have the SAME run ID
as the historical model. A matching run ID alone does not mean matching weights.

Run this from the repository root before preparation. It preserves the corpus
revision instead of resolving a newer corpus revision during the first run.
The upload intentionally does not force-overwrite an existing pins file.
The volume-create command assumes ir646-agentir does not exist. Skip only that
command if the volume already exists and is genuinely empty; do not reuse the
original experiment's populated volume for a fresh-training comparison.

```bash
python -m modal volume create ir646-agentir
python - <<'PY'
import json
from pathlib import Path

settings = json.loads(Path("experiment_config.json").read_text())
Path("/tmp/ir646_source_pins.json").write_text(
    json.dumps(settings["revisions"], indent=2)
)
PY
python -m modal volume put ir646-agentir /tmp/ir646_source_pins.json /agentir/sources/modal_source_pins.json
cd code
```

3. Run and review the bounded benchmark.

```bash
python -m modal run --detach modal_agentir.py --mode benchmark --precision bf16
```

The explicit --precision bf16 matters: the original CLI defaults to FP16.
Preparation downloads the pinned sources, then a bounded GPU benchmark measures
BF16 training and FP16 retrieval inference. Its temporary adapter and synthetic
search vectors measure throughput; they are not the final trained baseline.

Wait for the benchmark to finish with status complete and review its projection.
Save the printed run ID. If it reports incomplete, inspect its error before
attempting full training. Image building and CPU preparation take additional
time beyond the 600-second GPU benchmark.

4. Train to completion using the printed run ID.

```bash
IR646_RUN_ID="paste-the-run-id-printed-by-the-benchmark"
python -m modal run --detach modal_agentir.py --mode train --run-id "$IR646_RUN_ID" --seconds 25200
```

Wait for completion. A time-bounded invocation may report paused_at_time_limit;
repeat the same command with the same run ID to resume the latest complete
checkpoint, including optimizer/scheduler state. Run only one stage at a time.
Training precision comes from the prepared run manifest.

Successful full training writes training/final_adapter/ and
training/training_complete.json under the run directory. The expected completed
BF16 schedule has global step 1,310 and Trainer epoch 2.0. Partial checkpoints
and benchmark adapters are not accepted for the final baseline evaluation.

The cloud run directory is /data/agentir/experiments/RUN_ID. For Modal volume
CLI commands, omit the /data mount prefix and use /agentir/experiments/RUN_ID.
Detached work continues after disconnection once the cloud job has started;
reports persist in the Modal volume even when a local report is not written.

5. Re-encode the corpus, then replay the benchmark trajectories.

```bash
python -m modal run --detach modal_agentir.py --mode encode --run-id "$IR646_RUN_ID" --seconds 5400
```

If encoding pauses, repeat it until it reports complete. Encoding rebuilds
all 100,195 document embeddings with the trained encoder. Pretrained Qwen
document vectors are not interchangeable with this fine-tuned encoder's vectors.

After encoding completes, launch replay:

```bash
python -m modal run --detach modal_agentir.py --mode replay --run-id "$IR646_RUN_ID" --seconds 1800
```

If replay pauses, repeat it until complete. The wrappers prepare the required
data on CPU before GPU work.

Replay processes all 127,600 searches from the six released agents. It writes
question_metrics.csv and summary_by_agent.csv only after complete coverage.
The optional modal_agentir_postprocess.py controller was used for detached
postprocessing in the original workflow; the commands above use the main
wrapper and its original budget gates.

6. Download your newly trained adapter and result CSVs.

```bash
python -m modal volume get ir646-agentir "/agentir/experiments/$IR646_RUN_ID/training/final_adapter" ../retrained_adapter
python -m modal volume ls ir646-agentir "/agentir/experiments/$IR646_RUN_ID/replays"
```

Copy the replay directory name from that listing; it depends on the actual
adapter weights. Do not substitute the historical model's replay signature.

```bash
IR646_REPLAY_SIGNATURE="paste-the-replay-directory-name"
mkdir -p ../reproduced_results
python -m modal volume get ir646-agentir "/agentir/experiments/$IR646_RUN_ID/replays/$IR646_REPLAY_SIGNATURE/summary_by_agent.csv" ../reproduced_results/summary_by_agent.csv
python -m modal volume get ir646-agentir "/agentir/experiments/$IR646_RUN_ID/replays/$IR646_REPLAY_SIGNATURE/question_metrics.csv" ../reproduced_results/question_metrics.csv
```


USING THE TRAINED MODEL WITHOUT RETRAINING

model/final_adapter/ contains LoRA weights and their PEFT configuration. These
are trained adjustments to the base model, not a standalone full model.

    pinned Qwen3-Embedding-0.6B base + supplied adapter = our fine-tuned encoder

The adapter configuration retains an absolute path from the original Modal
container. Load the pinned base explicitly, then apply the local adapter as in
the example below. No Modal account is needed for this inference example.

The example targets the original A100-compatible Linux environment with Python
3.11 and a CUDA 12.6 development toolchain for FlashAttention. It follows the
FP16 CUDA inference path used in evaluation; it is not a CPU example. The
training launcher's requirements file installs only the Modal SDK.
For inference, install the following dependencies in a separate environment.
Run from the repository root:

```bash
python3.11 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install torch==2.7.1 transformers==4.53.2 peft==0.18.0 accelerate==1.10.1 huggingface_hub==0.36.0 numpy==1.26.4 faiss-cpu==1.11.0.post1
python -m pip install wheel==0.45.1 setuptools==80.9.0 packaging==25.0 ninja==1.11.1.4 psutil
python -m pip install flash-attn==2.8.3 --no-build-isolation
cd code
```

Run the following Python example with code/ as the working directory. It loads
the supplied adapter and ranks two illustrative documents. The repository's
query_text() and encode() helpers supply the exact instruction, truncation,
left padding, last-token pooling, and normalization used by the workflow.

```python
from pathlib import Path

import faiss
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from agentir_config import MODEL_ID, REVISIONS
from agentir_workload import encode, query_text

if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA GPU.")

revision = REVISIONS[MODEL_ID]
adapter = Path("../model/final_adapter").resolve()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=revision, padding_side="left"
)
tokenizer.truncation_side = "right"

base = AutoModel.from_pretrained(
    MODEL_ID,
    revision=revision,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)
model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
model = model.cuda().eval()
model.config.use_cache = False
model.requires_grad_(False)

# Replace these illustrative texts with your own document collection.
documents = [
    "LoRA adapts a model by training small low-rank parameter matrices.",
    "BM25 ranks documents using lexical term-matching statistics.",
]
docids = ["example-lora", "example-bm25"]

document_vectors = encode(
    model, tokenizer, documents, limit=4096, batch_size=4
)
# Match the historical corpus cache's FP16 round-trip and re-normalization.
document_vectors = np.ascontiguousarray(
    document_vectors.astype(np.float16), dtype=np.float32
)
faiss.normalize_L2(document_vectors)
index = faiss.IndexFlatIP(1024)
index.add(document_vectors)

step = {
    "reasoning": "I need an explanation of parameter-efficient adaptation.",
    "query": "How does LoRA adapt a pretrained model?",
}
text, reasoning_was_trimmed = query_text(step, tokenizer, limit=8192)
query_vectors = encode(
    model, tokenizer, [text], limit=8192, batch_size=4, is_query=True
)
scores, positions = index.search(query_vectors, min(10, len(docids)))
for position, score in zip(positions[0], scores[0]):
    if position >= 0:
        print(docids[int(position)], float(score))
```

Document text is encoded without a query instruction. A query combines the
current reasoning and current search query with the training instruction.
Embeddings are normalized so inner product is cosine similarity. This is an
embedding/retrieval model; do not use a text-generation pipeline to query it.

The demo ranks its two example documents. To reproduce BrowseComp-Plus scores,
use the pinned full corpus and trajectories, the trained model's document
index, and the replay/evaluation logic. A small custom collection will not
produce the saved benchmark scores.


RESULTS AND REPOSITORY FILES

  results/summary_by_agent_percent.csv   Report-ready scores in percent
  results/summary_by_agent.csv           Original scores on a 0-to-1 scale
  results/question_metrics.csv          4,980 rows: 6 agents x 830 questions

Metrics are Evidence Recall@5, Evidence nDCG@10, Gold Recall@5, and trajectory-
union evidence Recall@5. The first three average searches within each question,
then question scores within each agent; zero-search questions are excluded.
Union recall uses the top five from every search and includes zero-search
questions as zero. Evaluation is fixed offline replay, not final-answer
accuracy or a live agent run. BrowseComp-Plus labels are used for evaluation,
not for retriever training or model selection.

  code/browsecomp_dr_synth_trained_qwen06_replay.ipynb
      Original Modal controller notebook. Open with code/ as the working
      directory. Its saved progress notes are historical; follow this README
      for the completed baseline and explicit BF16 commands. Launch switches
      are off by default, and it has no saved remote-training outputs.
  code/modal_agentir.py
      Modal image, CPU preparation, benchmark, and stage entrypoints.
  code/modal_agentir_postprocess.py
      Original detached encoding/replay controller.
  code/agentir_config.py, code/agentir_deepspeed.json
      Defaults, pinned dependencies/revisions, and DeepSpeed configuration.
  code/agentir_data.py
      Training preprocessing and current-reasoning trajectory parsing.
  code/agentir_training.py
      Tevatron/DeepSpeed training, checkpoints, and adapter saving.
  code/agentir_workload.py
      Encoder loading, corpus encoding, exact search, and evaluation metrics.
  code/requirements-modal.txt
      Local Modal SDK pin. ML dependencies are installed in the remote image.
  experiment_config.json
      Completed BF16-stage settings, inference settings, and source revisions.
  model/final_adapter/
      Final trained LoRA weights and PEFT configuration; base weights separate.

Keep the original source modules together. Their hashes are checked against
prepared run manifests, so source edits require preparation of a compatible
experiment. This repository omits training logs, historical recovery files,
the old Colab notebook, virtual environments, caches, the full corpus/index,
and raw trajectories.


OFFICIAL API REFERENCES

PEFT 0.18.0 model loading and adapter handling:
https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model

Pinned Qwen3-Embedding-0.6B model card:
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md

Modal detached runs and volume commands:
https://modal.com/docs/cli/latest/run
https://modal.com/docs/cli/latest/volume
