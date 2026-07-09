# MemoryBank On-Device Reproduction Notes

This document describes the local MemoryBank reproduction path used by
StateBudgetMem. The goal is to reproduce the MemoryBank paper's long-term
memory mechanics as a baseline, not to claim a full SiliconFriend reproduction.

## Paper Mechanism Mapping

MemoryBank paper mechanism:

- Each memory starts with memory strength `S = 1`.
- When a memory is recalled, `S += 1`.
- Recall also resets or updates the last accessed time.
- Retention follows an Ebbinghaus-style curve: `R = exp(-t / S)`.
- When retention is below a threshold, the memory may be treated as forgotten.

Current project mapping:

- `MemoryPiece.strength` stores `S` and starts at `1.0`.
- `MemoryBank.retrieve()` applies the spacing effect by increasing
  `strength` and updating `last_accessed`.
- `MemoryBank.update_forgetting()` keeps the existing legacy return shape.
- `MemoryBank.update_forgetting_with_log()` runs the same forgetting update and
  returns machine-readable retention events.
- `MemoryBank.forgetting_log()` computes retention events without mutating the
  memory bank.

## Reproduced Parts

- Local memory storage and FAISS retrieval.
- Memory-augmented prompt construction from retrieved memories, global user
  portrait, global event summary, and the current user query.
- Strength reinforcement after recall.
- `last_accessed` update after recall.
- Ebbinghaus-style retention calculation.
- Forgotten-memory identification using a configurable threshold.
- JSONL/JSON logs for plotting and later evaluation.

## Approximate Parts

- The paper's memory update model is exploratory; this project uses the
  simplified implementation already present in `MemoryBank`.
- The demo runner uses a deterministic local hash embedding model to avoid
  downloading sentence-transformers during a local smoke run.
- The demo does not reproduce SiliconFriend psychological-dialog fine-tuning.
- `build_augmented_prompt()` currently constructs the prompt and exposes the
  retrieved context for retrieval-level evaluation. It does not call an LLM by
  itself; `MemoryAugmentedAgent` can pass the prompt to an injected local model
  or a template caller.

## Memory-Augmented Prompt

`MemoryBank.build_augmented_prompt()` follows the paper's prompt composition
order used in the local reproduction:

```text
relevant memories
+ global user portrait
+ global event summary
+ current user query
```

The method returns both the rendered `prompt_template` and structured fields:

- `relevant_memories`: formatted top-k retrieved memories.
- `global_user_portrait`: current global user portrait.
- `global_event_summary`: current global event summary.
- `current_user_query`: the raw user query.
- `prompt_sections`: the four prompt sections in machine-readable form.
- `retrieved_memory_ids`: IDs used in the prompt.
- `retrieved_memories`: raw retrieval rows for retrieval-only evaluation.

For the first reproduction stage, evaluate retrieval quality and prompt
composition without invoking a cloud LLM. For the second stage, inject a local
small model or deterministic template through `MemoryAugmentedAgent`.

## Run The Demo

Install MemoryBank optional dependencies first:

```bash
python -m pip install -e ".[memorybank]"
```

Run the formal local/on-device reproduction script:

```bash
python tools/memorybank/run_ondevice_reproduction.py
```

It writes:

```text
results/memorybank/ondevice/raw/*.jsonl
results/memorybank/ondevice/summaries/*.json
results/memorybank/ondevice/resources/*.json
```

Until the formal MemoryBank-style dataset is ready, this script uses the
built-in paper-aligned sample from `default_paper_storage_spec()`. The output
format is already stable: TODO2 can later add a dataset loader without changing
the raw/summary/resource schema.

Run the MemoryBank on-device budget sweep:

```bash
python tools/memorybank/run_budget_sweep.py
```

Default sweep grid:

```text
top_k = 1, 3, 5
prompt_token_budget = 128, 256, 512, 1024
memory_count = 100, 500, 1000, 5000
forgetting_threshold = 0.1, 0.3, 0.5
```

The sweep writes:

```text
results/memorybank/budget_sweep/raw/*.jsonl
results/memorybank/budget_sweep/summaries/*.json
results/memorybank/budget_sweep/resources/*.json
```

For a fast smoke run:

```bash
python tools/memorybank/run_budget_sweep.py --quick
```

The sweep uses deterministic synthetic memories with controlled relevant,
stale, and filler memories. This lets us test whether tighter top-k, token,
storage, and forgetting budgets make MemoryBank lose relevant memories or
retrieve stale memories before the formal TODO2 dataset is ready.

Run the local forgetting/reinforcement demo:

```bash
python tools/memorybank/run_memorybank_forgetting_demo.py
```

Outputs are written under:

```text
results/memorybank/forgetting_demo/
```

The generated files are:

```text
memorybank_reinforcement_log.jsonl
memorybank_forgetting_log.jsonl
memorybank_forgetting_summary.json
```

## Field Meanings

`memorybank_reinforcement_log.jsonl`:

- `memory_id`: recalled memory ID.
- `query`: query that recalled the memory.
- `before_strength`: strength before recall.
- `after_strength`: strength after recall.
- `before_last_accessed`: last access time before recall.
- `after_last_accessed`: last access time after recall.
- `retrieval_rank`: rank after composite-score sorting.
- `retrieval_score`: score used for final ranking.
- `semantic_score`: embedding similarity score.
- `composite_score`: semantic score combined with strength and time decay.
- `timestamp`: recall timestamp.

`memorybank_forgetting_log.jsonl`:

- `memory_id`: memory ID.
- `strength`: strength used to compute retention.
- `last_accessed`: last access timestamp.
- `elapsed_hours`: hours since last access.
- `retention`: `exp(-elapsed_hours / strength)`.
- `is_forgotten`: whether retention is below the threshold.
- `threshold`: forgetting threshold.

`memorybank_forgetting_summary.json`:

- `forgotten_memory_ids`: list of memories below threshold.
- `reinforcement_event_count`: number of recall log rows.
- `forgetting_event_count`: number of retention log rows.
- `memory_stats`: MemoryBank storage and index stats after the run.

`results/memorybank/ondevice/raw/*.jsonl`:

- `query_id`: deterministic query row ID.
- `query`: user query used for retrieval.
- `retrieved_memory_ids`: memories inserted into the prompt.
- `retrieved_memories`: raw MemoryBank retrieval rows.
- `prompt_sections`: relevant memories, global user portrait, global event
  summary, and current user query.
- `prompt_template`: rendered MemoryBank-style prompt.
- `latency_ms`: retrieval plus prompt-construction latency.
- `prompt_token_estimate`: deterministic local token proxy.
- `index_size`: number of FAISS-indexed memories.

`results/memorybank/ondevice/summaries/*.json` records query count, mean
latency, mean prompt token estimate, output paths, and the three-layer storage
report.

`results/memorybank/ondevice/resources/*.json` records local-only execution,
hardware/runtime information, peak traced memory, index size, memory count, and
output file sizes.

## Metrics

The on-device reproduction runner records the three metrics used in the
MemoryBank paper and extra on-device metrics.

Paper metrics:

- `memory_retrieval_accuracy`: approximates the paper's Memory Retrieval
  Accuracy with deterministic keyword labels over retrieved memories.
- `response_correctness`: approximates Response Correctness with deterministic
  keyword labels over a local template answer.
- `contextual_coherence`: approximates Contextual Coherence with a deterministic
  grounding heuristic that checks whether the answer is connected to the query
  and retrieved memories.

On-device metrics:

- `retrieval_latency_ms`: retrieval plus prompt-construction latency.
- `faiss_index_size`: number of indexed memories.
- `storage_size_bytes`: estimated local memory text and metadata size.
- `prompt_token_cost`: deterministic prompt token proxy.
- `peak_tracemalloc_bytes`: peak traced Python memory during the runner.
- `stale_retrieval_rate`: keyword-labeled stale retrieval rate. It is 0 for the
  built-in smoke sample unless stale labels are provided.
- `relevant_loss_rate`: budget-sweep rate of cases where relevant memory is not
  fully preserved after retrieval and prompt-budget selection.
- `stale_retrieval_case_rate`: budget-sweep rate of cases where at least one
  stale memory is selected.

These paper metrics are first-stage proxies because the formal dataset and
human/LLM judge labels are not ready yet. After TODO2 adds a MemoryBank-style
dataset, the same metric functions can consume dataset-level labels instead of
the built-in keyword spec.

## Tests

Run:

```bash
python -m pytest tests/baselines/memorybank/test_system.py -q
python -m pytest tests/baselines/memorybank/test_ondevice_reproduction.py -q
python -m pytest tests/baselines/memorybank/test_metrics.py -q
python -m pytest tests/baselines/memorybank/test_budget_sweep.py -q
```

The deterministic tests avoid real waiting and do not require downloading an
embedding model.
