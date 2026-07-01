# StateBudgetMem

**资源受限端侧个人智能体的时态一致性长期记忆管理**  
*Temporally Consistent Long-Term Memory for Resource-Constrained On-Device Agents*

StateBudgetMem studies how a long-running personal agent can keep the current
user state correct, preserve historical versions, avoid stale-memory misuse,
and retrieve useful memories under storage and context-token budgets.

## Current status (2026-07 updated)

| Stage | Status |
|-------|--------|
| Deterministic TF-IDF & MemoryBank baselines | ✅ |
| Controlled datasets (44 scenarios) | ✅ |
| State versioning engine (ADD/SUPERSEDE/TEMP_INVALIDATE/RESTORE) | ✅ |
| Current View & History View | ✅ |
| Query routing (rule + LLM, 4 types: CURRENT/HISTORICAL/CHANGE/GENERAL) | ✅ |
| End-to-end pipeline (query → route → view → retrieve) | ✅ |
| Memora dataset evaluation (10 personas, monthly/weekly) | ✅ |
| Staleness-aware retrieval & budget selection | 🟡 next |
| LLM answer generation | 🟡 next |

### Test coverage: **226 passed** (pytest -q, 6.5s)

## Quick start

```bash
# Install
pip install -e ".[test]"

# Run all tests
pytest -q

# Pipeline: route a query and retrieve relevant memories (offline)
statebudgetmem pipeline --query "我现在还喜欢吃辣吗？" -v

# Pipeline with LLM routing
statebudgetmem pipeline --query "我现在还喜欢吃辣吗？" --mode llm \
  --api-key "sk-xxx" --base-url "https://api.deepseek.com" --model "deepseek-chat"

# Evaluate routing on all 10 Memora personas
python tools/routing/eval_on_memora.py --memora-dir Memora --all-personas

# With LLM routing (better accuracy on English Memora questions)
python tools/routing/eval_on_memora.py --memora-dir Memora --all-personas \
  --mode llm --model "deepseek-chat"
```

## Architecture

```text
user query
    │
    ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  routing/    │───▶│    views/        │───▶│  retrieval/  │
│ QueryRouter  │    │ MemoryViewManager│    │ TfidfRetriever│
│ 4-type class │    │ current+history  │    │   top-k      │
└──────────────┘    └────────┬────────┘    └──────┬───────┘
                             │                    │
                             ▼                    ▼
                    ┌────────────────┐    ┌──────────────┐
                    │  versioning/   │    │   context    │
                    │VersioningEngine│    │  (→ answer)  │
                    │ SUPERSEDE/     │    └──────────────┘
                    │ RESTORE/DELETE │
                    └────────────────┘
```

## Module map

| Module | Location | Description |
|--------|----------|-------------|
| `schemas/` | `src/statebudgetmem/schemas/` | MemoryRecord, QueryRecord, Scenario (pydantic v2) |
| `core/` | `src/statebudgetmem/core/` | Abstract interfaces: MemorySystem, VersionManager, etc. |
| `versioning/` | `src/statebudgetmem/versioning/` | Version graph, SUPERSEDE/RESTORE, current/history resolution |
| **`routing/`** | **`src/statebudgetmem/routing/`** | **LLM + rule-based query classification (your module)** |
| `views/` | `src/statebudgetmem/views/` | Current View + History View built from versioning |
| `retrieval/` | `src/statebudgetmem/retrieval/` | Shared TF-IDF retriever |
| `baselines/` | `src/statebudgetmem/baselines/` | MemoryBank, TF-IDF baselines, agents, evaluators |
| `apps/` | `src/statebudgetmem/apps/` | End-to-end MemoryPipeline |
| `evaluation/` | `src/statebudgetmem/evaluation/` | Shared metrics (Recall@K, etc.) |

## Main commands

```bash
# Controlled TF-IDF baseline
statebudgetmem run --config configs/baseline.yaml

# Query routing (single query)
statebudgetmem route "我现在还喜欢吃辣吗？"

# Pipeline (end-to-end: route → view → retrieve)
statebudgetmem pipeline --query "我现在适合吃什么？" -v
statebudgetmem pipeline --query "我现在适合吃什么？" --mode llm --model "deepseek-chat"

# MemoryBank evaluation
statebudgetmem evaluate-memorybank --output results/memorybank_evaluation.json
statebudgetmem analyze-staleness --backend tfidf

# Memora routing evaluation
python tools/routing/eval_on_memora.py --memora-dir Memora --persona software_engineer
python tools/routing/eval_on_memora.py --memora-dir Memora --all-personas --mode llm --model "deepseek-chat"

# Debug routing prompts
python tools/routing/debug_routing.py --dry-run --query "我现在还喜欢吃辣吗？"

# Gradio visual comparison (MemoryBank)
statebudgetmem-demo
```
