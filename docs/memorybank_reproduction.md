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

## Run The Demo

Install MemoryBank optional dependencies first:

```bash
python -m pip install -e ".[memorybank]"
```

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

## Tests

Run:

```bash
python -m pytest tests/baselines/memorybank/test_system.py -q
```

The deterministic tests avoid real waiting and do not require downloading an
embedding model.

