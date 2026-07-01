# Task001 Implementation Plan

## Scope

This task builds only the deterministic offline baseline and experiment skeleton:

- Python project structure under `src/statebudgetmem`.
- Pydantic data models for `MemoryRecord`, `QueryRecord`, and `Scenario`.
- A small controlled JSONL dataset with at least 12 state-change scenarios.
- Offline TF-IDF + cosine similarity Top-K retrieval.
- Retrieval metrics: Recall@K, Valid Recall@K, Stale Retrieval Rate, average retrieved token cost, and retrieval latency.
- CLI entry point that writes raw JSONL plus summary JSON and CSV files under `results/`.
- Pytest coverage for models, retrieval, metrics, and CLI execution.
- README usage notes for this baseline.

## Non-Scope

This task intentionally does not implement:

- LLM calls or external APIs.
- Dense embeddings or vector databases.
- Version update operations.
- Current-state or historical-version views.
- Query routing.
- Token budget optimization or storage eviction.
- Any chat, web, or demo UI.

## Implementation Notes

- The repository initially contained only `AGENTS.md`, `README.md`, `docs/StateBudgetMem_research_plan.md`, and the task file.
- `pydantic` is available in the local Python environment; `scikit-learn` and `PyYAML` are not available.
- To keep the baseline fully offline, TF-IDF and cosine similarity will be implemented with the Python standard library.
- `configs/baseline.yaml` will use a small flat YAML subset parsed by project code, avoiding a runtime PyYAML dependency.
- The controlled data will include a known stale retrieval case where a semantically similar but stale spicy-food preference is retrieved by plain TF-IDF.

## Validation Plan

1. Run `pytest -q`.
2. Run `python -m statebudgetmem.cli run --config configs/baseline.yaml`.
3. Confirm generated files exist under `results/raw/` and `results/summaries/`.
4. Inspect raw results for at least one query with `stale_retrieval_rate > 0`, especially the spicy-food temporary health restriction case.
