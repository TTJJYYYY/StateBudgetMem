# Baselines Status

This document summarizes the baseline module status for the short-term project report.

## Implemented Baselines

### TF-IDF controlled baseline

Location:

- `src/statebudgetmem/baselines/tfidf/retriever.py`
- `src/statebudgetmem/baselines/tfidf/adapter.py`
- `src/statebudgetmem/baselines/tfidf/runner.py`

Status: implemented and offline runnable.

Capabilities:

- deterministic mixed Chinese/English tokenization;
- TF-IDF cosine retrieval;
- unified `MemoryMethod` adapter;
- controlled JSONL scenario loading;
- Recall@K, Valid Recall@K, Stale Retrieval Rate, token cost, and latency export;
- JSONL raw results and JSON/CSV summary results.

Primary command:

```bash
statebudgetmem run --config configs/baseline.yaml
```

### MemoryBank baseline

Location:

- `src/statebudgetmem/baselines/memorybank/system.py`
- `src/statebudgetmem/baselines/memorybank/agents.py`
- `src/statebudgetmem/baselines/memorybank/datasets.py`
- `src/statebudgetmem/baselines/memorybank/evaluator.py`
- `src/statebudgetmem/baselines/memorybank/staleness.py`
- `src/statebudgetmem/baselines/memorybank/demo.py`

Status: implemented as a baseline and demo module, with optional heavy dependencies.

Capabilities:

- FAISS + embedding semantic retrieval when optional dependencies are installed;
- lightweight `TFIDFMemoryBank` comparison backend;
- MemoryBank-style time decay and spacing-effect memory strengthening;
- MemoryBank vs stateless answer evaluation;
- built-in demo history and optional Memora loading;
- obsolete-memory analysis with OMR and COR;
- Gradio comparison demo entry point.

Optional install:

```bash
python -m pip install -e ".[memorybank]"
python -m pip install -e ".[demo]"
```

Primary commands:

```bash
statebudgetmem evaluate-memorybank --output results/memorybank/evaluation.json
statebudgetmem analyze-staleness --backend tfidf
statebudgetmem-demo
```

## Unified Baseline Comparison

Location:

- `src/statebudgetmem/baselines/runner.py`

Status: implemented as a small comparison runner for controlled retrieval experiments.

Supported methods:

- `tfidf_topk`
- `tfidf_memorybank`

Output metrics:

- Recall@K;
- Valid Recall@K;
- Stale Retrieval Rate;
- OMR, defined here as the stale retrieved-memory ratio;
- COR, defined here as the current/valid retrieved-memory ratio;
- average and total token cost;
- retrieval latency.

Command:

```bash
python -m statebudgetmem.baselines.runner \
  --dataset data/controlled/baseline_scenarios.jsonl \
  --results-dir results/baselines \
  --top-k 3
```

Outputs:

```text
results/baselines/raw/<run_id>.jsonl
results/baselines/summaries/<run_id>.json
results/baselines/summaries/<run_id>.csv
```

## Relationship To The Original main-copy Branch

The original `main-copy` branch provided the first MemoryBank reproduction idea and demo scaffold:

- MemoryBank storage and retrieval;
- FAISS semantic retrieval;
- time decay and spacing-effect strengthening;
- MemoryBank-augmented agent vs stateless agent comparison;
- built-in demo conversation data;
- Memora loading and evaluation prototype;
- obsolete-memory analysis with OMR/COR.

These ideas were integrated into the current `main` branch under `src/statebudgetmem/baselines/memorybank/`, then packaged, tested, and connected to the project CLI. The original branch should be treated as the prototype source rather than merged directly, because it also contains root-level scripts, local IDE files, and early hard-coded configuration.

## Remaining Limitations

- The FAISS MemoryBank path requires optional dependencies and is not part of the minimal offline test set.
- MemoryBank answer evaluation is still a baseline/demo comparison, not the final StateBudgetMem method.
- The unified baseline runner currently focuses on retrieval metrics. Final answer metrics and figure generation still need a later experiment task.
- Storage-budget control and stale-risk-aware selection are not yet implemented in baselines; they belong to the next method stage.
