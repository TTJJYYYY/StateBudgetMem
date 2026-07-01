# Architecture

## 1. Research pipeline

The intended final system has two flows.

### Memory write flow

```text
conversation
  → preprocessing
  → structured MemoryRecord / MemoryPiece
  → versioning operation
  → current/history view synchronization
  → storage and budget management
```

### Query flow

```text
query
  → routing (CURRENT / HISTORICAL / CHANGE / GENERAL)
  → current view / history view / both / none
  → retrieval
  → stale-risk and budget selection
  → answer generation and evaluation
```

## 2. Module boundaries

### Shared contracts

- `interfaces.py`: stable public facade used across modules.
- `core/`: abstract online and controlled-experiment protocols.
- `schemas/`: validated records and result objects.

Changes here affect multiple contributors and must be announced before editing.

### Baselines

Each baseline is self-contained:

```text
baselines/memorybank/
├── system.py       # FAISS MemoryBank + original TF-IDFMemoryBank
├── agents.py       # stateless and memory-augmented agent wrappers
├── datasets.py     # generic JSON, Memora, and built-in demo data
├── evaluator.py    # answer comparison, LLM judge, batch export
├── staleness.py    # obsolete-memory labels and OMR/COR analysis
└── demo.py         # full Gradio comparison

baselines/tfidf/
├── retriever.py    # deterministic TF-IDF cosine retrieval
├── adapter.py      # MemoryMethod adapter
└── runner.py       # controlled-data experiment and result export
```

A baseline-specific evaluator or demo belongs beside that baseline, not in the
shared `evaluation/` or final `apps/` package.

### Research method modules

- `preprocessing/`: turns raw dialogue into structured memory candidates.
- `versioning/`: classifies and applies ADD, MERGE, SUPERSEDE,
  TEMP_INVALIDATE, RESTORE, DELETE, and NOOP.
- `views/`: will construct current and historical memory views.
- `routing/`: classifies query intent and selects a view.
- `retrieval/`: contains shared protocols for future retrievers.
- `evaluation/`: contains metrics that apply to every method.
- `apps/`: reserved for the final integrated StateBudgetMem demo.

## 3. Query-to-view contract

```text
CURRENT     → ViewType.CURRENT
HISTORICAL  → ViewType.HISTORY
CHANGE      → ViewType.BOTH
GENERAL     → ViewType.NONE
```

`GENERAL` must not silently retrieve personal memory.

## 4. Public data contracts

Use `statebudgetmem.interfaces`; do not redefine these objects locally.

### Online layer

- `MemoryPiece`
- `MemorySystem`
- `MemoryType`
- `MemoryStatus`
- `UpdateOperation`
- `VersionManager`
- `ViewManager`
- `QueryRouter`
- `ViewType`

### Controlled-experiment layer

- `MemoryRecord`
- `QueryRecord`
- `MemoryMethod`
- `MethodResult`
- `QueryType`
- `RetrievedMemory`
- `Scenario`

## 5. Test organization

Tests mirror source ownership:

```text
src/statebudgetmem/baselines/memorybank/system.py
tests/baselines/memorybank/test_system.py

src/statebudgetmem/routing/router.py
tests/routing/test_router.py
```

Cross-module behavior belongs under `tests/integration/`.
