# StateBudgetMem

**资源受限端侧个人智能体的时态一致性长期记忆管理**  
*Temporally Consistent Long-Term Memory for Resource-Constrained On-Device Agents*

StateBudgetMem studies how a long-running personal agent can keep the current
user state correct, preserve historical versions, avoid stale-memory misuse,
and retrieve useful memories under storage and context-token budgets.

This repository is the cleaned, structured union of the former `main`,
`routing`, and `feature/tfidf-baseline-framework` branches. The original demo,
MemoryBank baseline, controlled datasets, previous results, routing code, and
versioning implementation are preserved; duplicate v1/v2 files and local IDE
artifacts are not.

## Current status

Implemented:

- deterministic TF-IDF controlled baseline and 44 controlled scenarios;
- MemoryBank/FAISS baseline, lightweight agents, Memora adapters, answer
  evaluation, stale-memory analysis, and full Gradio comparison demo;
- structured memory preprocessing;
- state-versioning engine and tests;
- rule-based and LLM query routing;
- shared schemas, interfaces, metrics, CLI, and preserved experiment outputs.

Next major stage:

- Current View and History View;
- a unified end-to-end pipeline;
- budget-aware selection and final StateBudgetMem visualization.

## Collaboration-oriented structure

```text
StateBudgetMem/
├── configs/                         # reproducible experiment configuration
├── data/
│   ├── controlled/                  # 12 baseline + 32 temporal scenarios
│   └── external/memora/             # optional external dataset instructions
├── docs/
│   ├── ARCHITECTURE.md              # module boundaries and data flow
│   ├── TEAM_WORKFLOW.md             # four-person collaboration rules
│   ├── MIGRATION_FROM_THREE_BRANCHES.md
│   └── baselines/MEMORYBANK_BASELINE.md
├── examples/                        # minimal public-API examples
├── tools/
│   ├── memorybank/                  # baseline-specific analysis entry points
│   └── routing/                     # prompt and real-API debugging tools
├── results/                         # preserved and newly generated outputs
├── tests/                           # mirrors the source-module structure
│   ├── baselines/memorybank/
│   ├── baselines/tfidf/
│   ├── evaluation/
│   ├── integration/
│   ├── routing/
│   ├── schemas/
│   └── versioning/
└── src/statebudgetmem/
    ├── interfaces.py                # single public contract facade
    ├── core/                        # shared online/experiment protocols
    ├── schemas/                     # MemoryRecord, QueryRecord, MethodResult
    ├── data/                        # controlled-data loading
    ├── preprocessing/               # dialogue → structured memory
    ├── baselines/
    │   ├── memorybank/              # system, agents, data, eval, staleness, demo
    │   └── tfidf/                   # retriever, adapter, controlled runner
    ├── versioning/                  # matching, operations, graph, resolver
    ├── views/                       # Current/History views — next stage
    ├── routing/                     # rule and LLM routers
    ├── retrieval/                   # shared Retriever/Embedder protocols
    ├── evaluation/                  # method-independent retrieval metrics
    ├── apps/                        # reserved for the final system demo
    └── cli.py
```

The key rule is: **method-specific code stays together; shared contracts and
metrics stay method-independent**. Tests mirror the source path, so a module and
its tests are easy to locate.

## Shared interfaces

All modules use one public import path:

```python
from statebudgetmem.interfaces import (
    # Online memory-system layer
    MemoryPiece,
    MemorySystem,
    MemoryType,
    MemoryStatus,
    UpdateOperation,
    VersionManager,
    ViewManager,
    QueryRouter,
    ViewType,

    # Controlled-experiment layer
    MemoryRecord,
    QueryRecord,
    MemoryMethod,
    MethodResult,
    QueryType,
    RetrievedMemory,
    Scenario,
)
```

`MemoryPiece` / `MemorySystem` describe a live memory backend such as
MemoryBank. `MemoryRecord` / `QueryRecord` / `MemoryMethod` / `MethodResult`
describe reproducible controlled experiments. These layers are related but not
duplicates. Do not create private copies of these types inside a feature module.

## Install and verify

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
```

Optional components:

```bash
python -m pip install -e ".[memorybank]"  # FAISS + embeddings
python -m pip install -e ".[llm]"         # OpenAI-compatible APIs
python -m pip install -e ".[demo]"        # MemoryBank + Gradio
```

## Main commands

Controlled TF-IDF baseline:

```bash
statebudgetmem run --config configs/baseline.yaml
```

Query routing:

```bash
statebudgetmem route "我现在还喜欢吃辣吗？"
python tools/routing/debug_routing.py --dry-run --query "我的饮食习惯怎么变化的？"
```

MemoryBank evaluation and stale-memory analysis:

```bash
statebudgetmem evaluate-memorybank --output results/memorybank/evaluation.json
statebudgetmem analyze-staleness --backend tfidf

# Full original-style utilities:
python tools/memorybank/run_evaluation.py --output results/memorybank/evaluation.json
python tools/memorybank/analyze_staleness.py --mode demo
```

MemoryBank visual comparison:

```bash
statebudgetmem-demo
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before changing shared
interfaces, and use [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md) for team
coordination.
