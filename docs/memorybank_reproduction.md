# MemoryBank On-Device Reproduction Notes

This document describes the local MemoryBank reproduction path used by
StateBudgetMem. The goal is to reproduce the MemoryBank paper's long-term
memory mechanics as a baseline, not to claim a full SiliconFriend reproduction.

## Paper Mechanism Mapping

MemoryBank paper mechanism:

- Each memory starts with memory strength `S = 1`.
- When a memory is recalled, `S += 1`.
- Recall also resets or updates the last accessed time.
- Retention follows an Ebbinghaus-style curve: `R = exp(-t / S)`, where
  `t` is measured in configurable retention time units.
- When retention is below a threshold, the memory may be treated as forgotten.

Current project mapping:

- `MemoryPiece.strength` stores `S` and starts at `1.0`.
- Retrieval computes `R = exp(-t / S)` for each candidate after filters and
  before optional forgetting exclusion. By default,
  `decay_interval_hours=24.0`, so one `t` unit is one day.
- `MemoryBank.retrieve()` applies the spacing effect only to memories that
  survive filtering, survive optional forgetting exclusion, and enter final
  Top-K.
- `MemoryBank.update_forgetting()` keeps the existing legacy return shape.
- `MemoryBank.update_forgetting_with_log()` runs the same forgetting update and
  returns machine-readable retention events. Its `strength *= 0.5` update for
  forgotten memories is a legacy simplification in this project, not an
  original-paper requirement.
- `MemoryBank.forgetting_log()` computes retention events without mutating the
  memory bank.

The implementation has three parts: storage (`dialog`, `summary`, and portrait
state), retrieval (FAISS candidate search plus MemoryBank scoring), and updating
(recall reinforcement plus the simplified forgetting update).

## Reproduced Parts

- Local memory storage and FAISS retrieval.
- Memory-augmented prompt construction from retrieved memories, global user
  portrait, global event summary, and the current user query.
- Strength reinforcement after recall.
- `last_accessed` update after recall.
- Ebbinghaus-style retention calculation.
- Configurable retention time unit through `decay_interval_hours`; outputs
  record `retention_time_unit_hours`.
- Forgotten-memory identification using a configurable threshold.
- Optional hard exclusion with `--exclude-forgotten`.
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
- `prompt_token_estimate`: deterministic token proxy for the full prompt.
- `forgotten_memory_ids`: candidates below threshold after filters and before
  optional exclusion.
- `excluded_forgotten_memory_ids`: candidates removed because
  `exclude_forgotten=True`.
- `strength_before_after`, `last_accessed_before_after`, and
  `access_count_before_after`: recall reinforcement deltas for final Top-K.

Default retrieval uses `exclude_forgotten=False`: retention and `is_forgotten`
are recorded, but forgotten candidates can still rank and enter the prompt for
baseline compatibility. With `--exclude-forgotten`, candidates whose retention
is below the threshold do not enter final Top-K, do not enter the prompt, and
are not reinforced by that query. This hard exclusion is an optional
on-device/ablation strategy, not the only behavior specified by the original
MemoryBank paper.

Retention uses:

```text
elapsed_seconds = max(0, now - last_accessed)
elapsed_time_units = elapsed_seconds / decay_interval_sec
R = exp(-elapsed_time_units / S)
```

`elapsed_hours` is still recorded for readability, while
`retention_time_unit_hours` records the configured time unit. The default is
24 hours.

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

PowerShell one-line examples:

```bash
python tools/memorybank/run_ondevice_reproduction.py --run-id a1_default
python tools/memorybank/run_ondevice_reproduction.py --run-id a1_exclude --exclude-forgotten
python tools/memorybank/run_phase1_baseline.py --smoke --embedding-backend hash --run-id phase1_default
python tools/memorybank/run_phase1_baseline.py --smoke --embedding-backend hash --exclude-forgotten --run-id phase1_exclude
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
- `elapsed_time_units`: elapsed time divided by the configured retention unit.
- `retention_time_unit_hours`: retention time unit in hours; default `24.0`.
- `retention`: `exp(-elapsed_time_units / strength)`.
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
- `forgotten_memory_ids`: filtered candidates whose retention was below the
  threshold before optional exclusion.
- retrieved memory rows include `elapsed_hours`, `elapsed_time_units`, and
  `retention_time_unit_hours` for auditing the retention calculation.
- `excluded_forgotten_memory_ids`: forgotten candidates actually removed when
  hard exclusion is enabled.
- `candidate_count_before_forgetting` / `candidate_count_after_forgetting`:
  candidate counts around optional hard exclusion.
- `exclude_forgotten`: whether hard exclusion was enabled.
- `strength_before_after`, `last_accessed_before_after`, and
  `access_count_before_after`: mutation audit for final retrieved memories.
- `embedding_backend`: `hash` for smoke/CI or `sentence-transformer` for local
  semantic retrieval.
- `embedding_model`: `deterministic_hash_embedding` for hash mode; otherwise
  the sentence-transformer model name or local path.
- `index_size`: number of FAISS-indexed memories.

`results/memorybank/ondevice/summaries/*.json` records query count, mean
latency, mean prompt token estimate, output paths, and the three-layer storage
report.

`results/memorybank/ondevice/resources/*.json` records local-only execution,
hardware/runtime information, peak traced memory, index size, memory count, and
output file sizes. Hash embedding is deterministic, does not access the
network, and is recorded as `deterministic_hash_embedding`. Sentence-transformer
mode is the local semantic retrieval path; the default is `all-MiniLM-L6-v2`,
first use may download model files, and offline runs require a cached model or
local model directory. It is not a cloud LLM API call.

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

These paper metrics are first-stage proxies because template answers and
keyword checks are not the original paper's human evaluation. After TODO2 adds a
MemoryBank-style dataset, the same metric functions can consume dataset-level
labels instead of the built-in keyword spec.

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
