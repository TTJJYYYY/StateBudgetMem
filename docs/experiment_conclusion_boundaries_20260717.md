# Experiment Conclusion Boundaries - 2026-07-17

## 1. Purpose

This document defines which StateBudgetMem artifacts can support formal
claims, which artifacts are only analysis or demo material, and what the group
leader must approve before those artifacts enter the final report.

The current project direction remains:

1. Build a credible, local-only On-device MemoryBank Core Baseline.
2. Compare StateBudgetMem Versioning / Views / Routing against that baseline
   under the same endpoint experiment framework.
3. Keep answer generation and Local LLM work separate from retrieval-level
   conclusions until a dedicated answer-level pilot protocol is approved.

## 2. Formal Evidence

### MemoryBank Core Baseline

Can support these claims:

- The MemoryBank core memory-system path can run locally with local files,
  local embeddings, and local FAISS.
- The baseline records write, retrieval, storage, prompt-token proxy, and
  process resource metrics.
- The implementation is a core memory-system baseline, not a full
  conversational agent or a complete reproduction of the original MemoryBank
  paper.

Must not support these claims:

- Full MemoryBank paper reproduction.
- Local LLM conversational closed loop.
- User personality evolution, daily automatic summaries, or original-paper
  human evaluation.

### Fair Comparison

The fair comparison is the formal source for method comparison. Its scope is
retrieval-level comparison under shared data, shared embedding backend, shared
Top-K, shared token budget, and shared runner settings.

Current reviewed result summary from the 96-query temporal challenge run:

| Method | Recall@K | Valid Recall@K | Stale Retrieval Rate | Token Proxy | Retrieval ms |
|---|---:|---:|---:|---:|---:|
| `memorybank_core` | 0.5427 | 0.5391 | 0.1701 | 32.00 | 21.46 |
| `memorybank_versioning` | 0.4394 | 0.6970 | 0.0035 | 28.88 | 21.80 |
| `memorybank_dual_views` | 0.5427 | 0.5391 | 0.1701 | 32.00 | 21.14 |
| `statebudgetmem_rule` | 0.2506 | 0.3455 | 0.0035 | 14.84 | 10.20 |
| `statebudgetmem_oracle` | 0.4758 | 0.7335 | 0.0035 | 29.29 | 21.33 |

Allowed formal claims:

- `memorybank_versioning` reduces stale retrieval relative to
  `memorybank_core` in the reviewed temporal challenge run.
- `memorybank_versioning` improves Valid Recall@K relative to
  `memorybank_core` in the reviewed temporal challenge run.
- `statebudgetmem_oracle` is an upper-bound diagnostic for routing and view
  selection; it is not deployable.
- `statebudgetmem_rule` is low-cost and low-stale but currently loses too much
  recall, so it needs error analysis before being presented as a strong final
  method.

Required before final report:

- Merge or otherwise preserve the machine-readable fair-comparison result files
  in the official repository history.
- Add query-type grouping for `CURRENT`, `HISTORICAL`, `CHANGE`, and, if
  present, `GENERAL`.
- Explain why `memorybank_dual_views` matches `memorybank_core` in the current
  MemoryBank fair-comparison configuration.

### Budget Sweep

Budget sweep is now merged as PR #5 and can support resource-trend claims.

Committed evidence:

- `results/budget_sweep/budget_sweep_rows.csv`
- `results/budget_sweep/budget_sweep_rows.json`
- `results/budget_sweep/budget_sweep_summary.csv`
- `results/budget_sweep/budget_sweep_summary.json`
- `results/budget_sweep/resource_metrics.json`
- `results/budget_sweep/manifest.json`
- `results/budget_sweep/figures/`
- `docs/budget_sweep_results_20260716.md`

Allowed claims:

- The sweep covers `token_budget`, `top_k`, `candidate_k`, and `memory_count`.
- It produced 1296 compact records over 144 legal configurations.
- It records resource metrics, artifact checksums, and five figures.
- It is local-only and does not call an LLM or cloud API.
- It is useful for quality-resource trend analysis.

Must not support these claims:

- It is not a formal MemoryBank-vs-StateBudgetMem method comparison.
- It does not replace the 96-query fair comparison.
- Its token metric is not a real tokenizer count.
- Its Hash Embedding result should not be read as MiniLM semantic retrieval
  performance.
- Its three synthetic probes are not enough for final method superiority
  claims.

### Reproduction Baseline Resource Statistics

The reproduction baseline is useful as endpoint resource evidence, but the
result files still need to be merged or archived in the official report path
before final submission.

Allowed claims after result preservation:

- The reproduction dataset contains 5 users, 35 user-days, 150 memory sources,
  and 50 probing questions.
- The run is local-only, uses MiniLM + FAISS, and does not call a cloud API.
- Resource values can be used to illustrate endpoint feasibility.

Must not support these claims:

- It is not a method superiority result.
- It is not an answer-generation benchmark.
- It is not a large-scale user study.

## 3. Demo And Analysis Artifacts

### MemoryExplorer

MemoryExplorer is an analysis and presentation tool. It can show:

- temporal memory status;
- current vs historical interpretation;
- retrieval differences;
- resource signals;
- answer citations in the showcase case.

It must not replace:

- fair comparison;
- budget sweep machine-readable results;
- retrieval metrics;
- answer-level evaluation.

### Minimal Demo / Case Entry

The minimal demo and Case Entry are useful for explaining the research problem.
They must be labeled as demo-only entry cases.

They must not support:

- formal accuracy claims;
- stale retrieval rate claims;
- final method superiority claims.

### Template Answer

`TemplateAnswerer` remains the default answer path.

Allowed role:

- deterministic demo and showcase answer path;
- stable local test path;
- placeholder for answer-level pilot design.

Boundary:

- Template Answer does not affect retrieval-level metrics.
- Template Answer must not be described as LLM generation quality.
- Template Answer must not be used as final answer accuracy evidence.

### Local LLM Answerer

`LocalLLMAnswerer` is an optional localhost Ollama pilot interface only.

Current status:

- Not part of formal fair comparison.
- Not part of budget sweep.
- Not required for final demonstration.
- No model installation or download should be done without leader approval.

## 4. Current Risks

| Risk | Impact | Owner | Required Action |
|---|---|---|---|
| Fair-comparison result files not yet merged into current clean main | Final report may cite untracked local artifacts | Leader + B | Merge or archive machine-readable fair-comparison outputs |
| `statebudgetmem_rule` recall is low | Proposed method may look too conservative | B | Query-type grouped error analysis |
| `memorybank_dual_views` equals core in current fair comparison | Dual-view contribution may be unclear | B | Explain current configuration and propose fix or wording |
| Budget sweep uses Hash Embedding and 3 probes | Overclaiming risk | Leader + A | Keep as resource trend only |
| Local LLM not installed or validated | Demo instability and conclusion pollution | Leader | Keep disabled unless explicitly approved |
| README and some docs show Chinese encoding issues in PowerShell reads | Reporting polish risk | Leader + C | Verify UTF-8 rendering in editor/browser before final |

## 5. Report Wording Rules

Use:

- "Current formal retrieval comparison shows..."
- "In the reviewed 96-query temporal challenge run..."
- "Budget sweep suggests resource trends under synthetic hash-embedding probes."
- "MemoryExplorer illustrates the mechanism and helps inspect examples."
- "Local LLM is reserved for a pilot and is not part of formal results."

Avoid:

- "StateBudgetMem has finally beaten MemoryBank."
- "We fully reproduced the MemoryBank paper."
- "Budget sweep proves the best production configuration."
- "Template Answer shows answer-generation quality."
- "Local LLM results are formal experimental results."

## 6. Leader Approval Gates

The following changes require leader approval before implementation or merge:

- installing Ollama or downloading any local LLM model;
- changing the default answerer away from `TemplateAnswerer`;
- changing formal fair-comparison settings based on budget sweep alone;
- merging answer-level results into formal conclusions;
- editing shared interfaces or schemas;
- deleting or rewriting existing result artifacts.

