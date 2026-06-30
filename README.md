# StateBudgetMem

Temporally consistent long-term memory management for resource-constrained on-device personal agents.

## Task001 Baseline

This stage provides a deterministic offline baseline for controlled memory retrieval experiments. It does not call external APIs, does not use LLMs, and does not implement version management, dual views, query routing, or budget optimization.

## Environment

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[test]"
```

The current baseline runtime dependency is `pydantic`. The TF-IDF + cosine retriever is implemented locally with the Python standard library so the experiment can run offline.

## Data Format

Controlled scenarios are stored in:

```text
data/controlled/baseline_scenarios.jsonl
```

Each JSONL line is one scenario with:

- `scenario_id`
- `description`
- `memories`
- `queries`

Queries include gold relevant, valid, and stale memory IDs for retrieval evaluation.

## Run Baseline

```bash
python -m statebudgetmem.cli run --config configs/baseline.yaml
```

Outputs are written to:

```text
results/raw/<run_id>.jsonl
results/summaries/<run_id>.json
results/summaries/<run_id>.csv
```

## Run Tests

```bash
pytest -q
```

## Current Limits

- Only deterministic flat TF-IDF Top-K retrieval is implemented.
- No LLM answer generation or answer accuracy metrics are included.
- No state version update logic is implemented.
- No current-state view, historical-version view, query router, token budget optimizer, storage eviction, or UI is included.
- The controlled data validates the experiment pipeline and should not be treated as final research evidence.
