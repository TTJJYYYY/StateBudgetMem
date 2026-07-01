# MemoryBank baseline

This package preserves the substantive content of the former `main` branch and
groups all MemoryBank-specific work in one place.

## Package layout

```text
src/statebudgetmem/baselines/memorybank/
├── system.py       # FAISS MemoryBank + original TFIDFMemoryBank
├── agents.py       # BaselineAgent and MemoryAugmentedAgent
├── datasets.py     # generic JSON, Memora, 36-turn demo history, 12 probes
├── evaluator.py    # keyword/LLM scoring, exports, persona batch evaluation
├── staleness.py    # obsolete detection and OMR/COR analysis
├── demo.py         # complete Gradio comparison with offline presets
└── __init__.py     # stable package exports
```

Corresponding tests are under:

```text
tests/baselines/memorybank/
```

## Implemented mechanisms

- FAISS + sentence-transformers semantic retrieval;
- raw dialog, summary, and user-portrait memory;
- Ebbinghaus-style retention `R = exp(-t / S)`;
- spacing effect by strengthening recalled memories;
- standardized `MemorySystem` operations;
- stateless versus memory-augmented answer comparison;
- generic JSON and Memora adapters;
- deterministic keyword scoring and optional LLM-as-Judge;
- stale-memory OMR/COR analysis;
- full offline/online visual demonstration.

## Installation

```bash
python -m pip install -e ".[memorybank]"
python -m pip install -e ".[demo]"       # includes Gradio and LLM extras
```

## Commands

Visual comparison:

```bash
statebudgetmem-demo
```

Evaluation:

```bash
statebudgetmem evaluate-memorybank \
  --output results/memorybank/evaluation.json

python tools/memorybank/run_evaluation.py \
  --memora-dir /path/to/Memora/data \
  --period weekly \
  --all-personas \
  --online
```

Stale-memory analysis:

```bash
statebudgetmem analyze-staleness --backend tfidf
python tools/memorybank/analyze_staleness.py --mode demo
```

## Preserved original result summary

The former branch recorded a weekly Memora subset over five personas and 75
questions:

| Persona | Baseline | MemoryBank | Relative gain | MemoryBank correct |
|---|---:|---:|---:|---:|
| software_engineer | 0.080 | 0.333 | 316.7% | 2/15 |
| academic_researcher | 0.107 | 0.287 | 168.7% | 2/15 |
| business_executive | 0.087 | 0.173 | 100.0% | 0/15 |
| financial_analyst | 0.087 | 0.253 | 192.3% | 2/15 |
| startup_founder | 0.100 | 0.127 | 26.7% | 0/15 |
| **Overall** | **0.092** | **0.235** | **155.1%** | **6/75** |

This is a preserved historical result, not a newly rerun claim. The previous
analysis identified stale-memory handling as a major limitation, which motivates
StateBudgetMem's versioning, dual-view, routing, and budget-aware design.
