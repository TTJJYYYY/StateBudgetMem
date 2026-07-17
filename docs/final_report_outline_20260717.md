# Final Report Outline - 2026-07-17

## 1. Report Positioning

StateBudgetMem is a research prototype for temporally consistent long-term
memory under endpoint resource constraints.

The final report should present two phases:

1. On-device MemoryBank Core Baseline: local memory storage, local embedding,
   local FAISS retrieval, resource recording, and reproducible baseline runs.
2. StateBudgetMem comparison: Versioning / Views / Routing components evaluated
   against MemoryBank core under the same runner and resource budget.

The report should not present the project as a general chat app, a complete
MemoryBank paper reproduction, or a finished local LLM agent.

## 2. Suggested Structure

### 2.1 Problem

Explain the central conflict:

- flat long-term memory can retrieve stale but semantically similar records;
- keeping only current state loses history;
- endpoint devices have limited context budget, storage, latency, and memory.

Main research question:

> Under the same endpoint resource budget, can version-aware memory management
> reduce stale-memory retrieval while preserving current and historical memory
> usefulness?

### 2.2 System Boundary

Describe the components:

- `MemoryRecord` and `QueryRecord` structured records;
- MemoryBank Core Baseline;
- Versioning;
- Current and History Views;
- Routing;
- shared dense retrieval through MemoryBank FAISS;
- Template Answer and Local LLM pilot boundary.

Make clear:

- MemoryBank Core Baseline is the baseline memory system.
- StateBudgetMem modules are proposed additions.
- Fair comparison is the formal method-comparison source.
- Budget sweep is resource-trend evidence.
- MemoryExplorer is a showcase and analysis tool.
- Local LLM is pilot only.

### 2.3 Data And Experimental Settings

Include:

- controlled temporal challenge for fair comparison;
- reproduction dataset for endpoint MemoryBank resource run;
- synthetic budget probes for budget sweep.

For each dataset, state what it is for and what it is not for.

| Dataset / Artifact | Main Use | Not For |
|---|---|---|
| temporal challenge fair comparison | formal retrieval-level method comparison | local LLM answer quality |
| reproduction MemoryBank run | endpoint baseline resource evidence | method superiority |
| budget sweep synthetic probes | quality-resource trend analysis | formal MemoryBank-vs-StateBudgetMem conclusion |
| minimal demo / Case Entry | explanation and presentation | formal metrics |

### 2.4 Main Results

#### Fair Comparison

Use the 96-query temporal challenge table as the formal method-comparison
evidence after the result files are preserved in the official branch.

Core observation:

- `memorybank_core` has higher overall Recall@K but much higher stale retrieval.
- `memorybank_versioning` lowers stale retrieval and improves Valid Recall@K.
- `statebudgetmem_oracle` shows the upper-bound value of correct routing.
- `statebudgetmem_rule` needs routing/error analysis before strong claims.
- `memorybank_dual_views` currently matches `memorybank_core` in the MemoryBank
  fair comparison and needs explanation.

#### Budget Sweep

Use PR #5 results to show endpoint resource tradeoffs:

- 1296 records;
- 144 legal configurations;
- dimensions: `token_budget`, `top_k`, `candidate_k`, `memory_count`;
- local-only;
- Hash Embedding;
- token proxy, not true tokenizer;
- five figures and manifest checksums.

Suggested wording:

> Budget sweep is used to understand resource trends. It does not replace the
> formal 96-query fair comparison and does not decide the final method winner.

#### On-device Resource Baseline

Use reproduction baseline resource results after preserving the result files.

Suggested wording:

> The reproduction baseline demonstrates local execution and endpoint resource
> logging on the group reproduction dataset. It is not an answer-generation or
> method-superiority benchmark.

### 2.5 Demo And Showcase

Show:

- Case Entry for the spicy-food temporal conflict;
- MemoryExplorer for current/history/stale inspection;
- Experiment Dashboard for formal fair-comparison and budget metrics;
- local-only resource panel.

Boundary sentence:

> The showcase helps explain and inspect the mechanism. Formal conclusions come
> from the machine-readable experiment outputs.

### 2.6 Limitations

List:

- fair comparison still needs query-type grouped analysis;
- `statebudgetmem_rule` recall is currently low;
- `memorybank_dual_views` contribution needs clarification;
- token metrics are proxies;
- budget sweep uses Hash Embedding and three synthetic probes;
- Local LLM is not part of formal results;
- public large-scale datasets and answer-level evaluation remain future work.

### 2.7 Next Work

Prioritize:

1. query-type fair comparison;
2. routing error analysis;
3. final showcase polish;
4. final report and slides;
5. optional Local LLM pilot only after leader approval.

## 3. Slide Outline

1. Title and research question.
2. Why flat memory fails under temporal state changes.
3. Two-phase project route.
4. MemoryBank Core Baseline architecture.
5. StateBudgetMem proposed modules.
6. Fair comparison setup.
7. Fair comparison result table.
8. Query-type and error-analysis TODO.
9. Budget sweep setup and resource trend.
10. On-device baseline resource evidence.
11. Showcase: MemoryExplorer and dashboard.
12. Boundary: formal result vs demo vs pilot.
13. Limitations.
14. Next week plan.

## 4. Current Team Tasks

| Owner | Current Focus | Leader Checkpoint |
|---|---|---|
| Leader | final report, conclusion boundary, Local LLM decision, PR review | keep claims conservative |
| A | budget sweep | done; results merged in PR #5 |
| B | fair comparison by query type and error analysis | must deliver grouped metrics and dual-view explanation |
| C | final showcase polish | must separate demo, analysis, formal results |

## 5. Merge And Branch Guidance

Use independent branches:

- `feature/report-and-llm-plan` for this leader documentation.
- `feature/fair-comparison-analysis` for B.
- `feature/showcase-polish` for C.

Before merging any branch:

```powershell
D:\projects\memorybankproject\StateBudgetMem\.venv\Scripts\python.exe -m pytest -q --basetemp .tmp\pytest-<branch> -p no:cacheprovider
D:\projects\memorybankproject\StateBudgetMem\.venv\Scripts\python.exe -m statebudgetmem.cli --help
```

For documentation-only branches, tests are still useful because this project is
small enough that a full smoke check is cheap and catches accidental import or
packaging damage.

## 6. Leader Checklist Before Final Report

- [ ] Fair-comparison machine-readable outputs are preserved or merged.
- [ ] Query-type grouped metrics are available.
- [ ] `statebudgetmem_rule` recall drop has an explanation.
- [ ] `memorybank_dual_views` result has an explanation.
- [ ] Budget sweep is cited only as resource-trend evidence.
- [ ] Local LLM is labeled pilot only.
- [ ] Template Answer is not treated as LLM answer quality.
- [ ] Showcase labels demo, analysis, and formal results separately.
- [ ] README and report render Chinese text correctly.
- [ ] Full tests pass on the final branch.

