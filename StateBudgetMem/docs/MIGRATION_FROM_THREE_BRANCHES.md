# Migration from the three former branches

This file explains where the substantive contents of the downloaded `main`,
`routing`, and `feature/tfidf-baseline-framework` archives now live.

## Former `main`

| Original content | Current location |
|---|---|
| `memorybank_v2.py` and useful APIs from `memorybank.py` | `src/statebudgetmem/baselines/memorybank/system.py` |
| baseline and memory-augmented agents | `src/statebudgetmem/baselines/memorybank/agents.py` |
| generic JSON/Memora loaders and built-in 36-turn demo history | `src/statebudgetmem/baselines/memorybank/datasets.py` |
| `evaluation_v2.py` answer comparison and batch evaluation | `src/statebudgetmem/baselines/memorybank/evaluator.py` |
| `analyze_obsolete_v2.py` stale-memory analysis | `src/statebudgetmem/baselines/memorybank/staleness.py` |
| complete `demo.py` Gradio UI and offline presets | `src/statebudgetmem/baselines/memorybank/demo.py` |
| quick-start code | `examples/memorybank_quickstart.py` |
| executable analysis wrappers | `tools/memorybank/` |
| previous stale-analysis output | `results/memorybank/obsolete_analysis.json` |
| former root interface definitions | canonicalized in `interfaces.py` and `core/online.py` |

The older non-v2 evaluator and obsolete-analysis duplicates were removed after
their useful behavior was incorporated into the canonical modules.

## Former `routing`

| Original content | Current location |
|---|---|
| routers, models, prompts, config, module README | `src/statebudgetmem/routing/` |
| full routing tests | `tests/routing/test_router.py` |
| prompt debugger and real-API runner | `tools/routing/` |

The routing-private duplicate query type and duplicate root router were removed;
routing now uses the shared `QueryType` and `ViewType` contracts.

## Former `feature/tfidf-baseline-framework`

| Original content | Current location |
|---|---|
| deterministic retriever and method adapter | `src/statebudgetmem/baselines/tfidf/` |
| reproducible baseline runner | `src/statebudgetmem/baselines/tfidf/runner.py` |
| 12 baseline + 32 temporal scenarios | `data/controlled/` |
| previous raw and summary outputs | `results/raw/`, `results/summaries/` |
| schemas and controlled-data I/O | `src/statebudgetmem/schemas/`, `src/statebudgetmem/data/` |
| preprocessing | `src/statebudgetmem/preprocessing/` |
| method-independent metrics | `src/statebudgetmem/evaluation/metrics.py` |
| complete versioning implementation | `src/statebudgetmem/versioning/` |
| versioning and baseline tests | mirrored locations under `tests/` |
| research and implementation documents | `docs/` |

## Intentionally not preserved as separate files

- `.idea/`, `__pycache__/`, `.pyc`, and build caches;
- duplicate root/package copies of the same code;
- obsolete non-v2 scripts after behavior was merged;
- empty external dataset folders.

No controlled scenario, previous result artifact, complete MemoryBank demo,
routing utility, or versioning source file was intentionally discarded.
