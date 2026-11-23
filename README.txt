BASELINE 4 CODE AND RESULTS

Qwen3-Embedding-0.6B fine-tuned on DR-Synth for retrieval using raw current
reasoning plus the current query. Queries and documents use the same trained
encoder. Training used BF16; retrieval inference used FP16. Input caps are
8,192 tokens for reasoning/query and 4,096 tokens for documents.

CONTENTS

code/                      Original Modal notebook and eight implementation files
experiment_config.json     Actual model, data, BF16 stage, and evaluation settings
model/final_adapter/       Final LoRA weights and PEFT configuration
results/                   Final per-agent and per-question result CSVs

CODE TO KEEP

browsecomp_dr_synth_trained_qwen06_replay.ipynb
                            Original Modal workflow controller notebook
modal_agentir.py             Main Modal image, preparation, and stage entrypoints
modal_agentir_postprocess.py Detached corpus encoding and replay controller
agentir_config.py            Configuration, pinned revisions, run management
agentir_data.py              Training preprocessing and trajectory parsing
agentir_training.py          Tevatron/DeepSpeed training implementation
agentir_workload.py          Embeddings, corpus index, retrieval, and metrics
agentir_deepspeed.json       DeepSpeed configuration
requirements-modal.txt      Local Modal SDK dependency

Keep the source files together and unchanged when working with the existing
saved run. The original preparation CLI defaults to FP16; select --precision
bf16 when preparing a new BF16 experiment. The actual completed BF16 stage
settings and evaluation revisions are in experiment_config.json.

TRAINING NOTEBOOK

Open code/browsecomp_dr_synth_trained_qwen06_replay.ipynb with code/ as the
working directory. It is an unchanged copy of the active project notebook.
Its cells launch the Modal preparation, benchmark, training, encoding, and
replay stages through modal_agentir.py. The GPU training implementation is
agentir_training.py, which runs inside the Modal worker.

The notebook's saved status notes describe earlier work and contain references
to historical recovery files not included in this sharing package. Use this
README and experiment_config.json for the completed baseline settings, and
results/ for the final scores. Its saved code cells have no execution outputs;
they do not contain the remote training log. Launch switches are off by default.
The _legacy.ipynb notebook is the older Colab implementation and is omitted.

RESULTS

summary_by_agent_percent.csv  Report-ready scores in percent, three decimals
summary_by_agent.csv          Original scores on a 0-to-1 scale
question_metrics.csv         4,980 rows: six agents times 830 questions

Evaluation covers 127,600 recorded searches over 100,195 documents. Metrics
are Evidence Recall@5, Evidence nDCG@10, Gold Recall@5, and trajectory-union
evidence Recall@5. The first three average searches within each question, then
questions within each agent; zero-search questions are excluded. Union recall
uses the top five from every search and includes zero-search questions as zero.
These are offline retrieval metrics, with no final-answer accuracy evaluation.

USING THE TRAINED MODEL

Load Qwen/Qwen3-Embedding-0.6B at revision
97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, then apply model/final_adapter/ with
PEFT. Use the tokenizer from the same pinned base. The adapter configuration
retains the original Modal cache path, so explicitly load the pinned base first.
The adapter is supplied for reusing the model associated with these scores.
A fresh training run is a new experiment and need not yield identical scores.

requirements-modal.txt installs the local launcher only. CUDA/ML dependencies
are installed by the Modal image defined in code/modal_agentir.py. Viewing
the result CSVs requires no Modal account or GPU.

NOT REQUIRED FOR SHARING THIS BASELINE

The legacy Colab notebook, MODAL_TRAINING_CHANGES.md, query-only baseline PDF,
agentir_resume.py, agentir_resume_utils.py, test_agentir.py, virtual environment,
Python caches, Insomnia installer, failed-trial reports, training logs,
checkpoint provenance, and verification tools are omitted. Recovery utilities
would be needed to reconstruct the historical recovery process; they are not
needed to reuse the supplied final adapter or inspect the results.

Original project files and cloud artifacts have not been deleted or changed.
