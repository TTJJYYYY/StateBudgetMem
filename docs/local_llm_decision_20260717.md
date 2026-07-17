# Local LLM Decision - 2026-07-17

## 1. Decision

Do not include Local LLM results in formal StateBudgetMem conclusions for the
current phase.

`TemplateAnswerer` remains the default answer path for tests, minimal demo, and
final showcase.

`LocalLLMAnswerer` remains an interface and optional pilot path only. The team
must not install Ollama, download local models, change the default answerer, or
mix Local LLM outputs into formal experiments unless the group leader explicitly
approves that action.

## 2. Rationale

The current formal evidence is retrieval-level:

- Recall@K;
- Valid Recall@K;
- Stale Retrieval Rate;
- retrieval latency;
- token proxy;
- resource metrics.

Adding a real local LLM now would introduce model-specific generation variance,
model installation cost, tokenizer differences, and demo reliability risk.
Those variables would make the current MemoryBank-vs-StateBudgetMem comparison
harder to explain.

The project should first finish:

1. query-type grouped fair comparison;
2. `statebudgetmem_rule` error analysis;
3. `memorybank_dual_views` explanation;
4. final report and showcase boundary cleanup.

## 3. Current Approved State

Approved:

- Keep `TemplateAnswerer` as default.
- Keep `LocalLLMAnswerer` code as a localhost-only pilot interface.
- Keep fallback behavior from Local LLM to Template Answer in demo scripts.
- Record local-only metadata and `cloud_api_calls=0`.
- Discuss Local LLM as future answer-level pilot work.

Not approved:

- Installing Ollama as part of the normal project workflow.
- Downloading any local LLM model.
- Changing default demo or showcase commands to `--answerer local_llm`.
- Reporting Local LLM output as formal experiment evidence.
- Comparing Local LLM and Template Answer inside the current fair-comparison
  tables.

## 4. If The Leader Later Approves A Pilot

If a Local LLM pilot is approved later, it must use a separate protocol:

1. Create a separate branch, for example `feature/local-llm-pilot`.
2. Write a short pilot plan before running:
   - selected model;
   - hardware;
   - local endpoint;
   - prompt template;
   - answer metrics;
   - fallback behavior;
   - expected runtime.
3. Use only `localhost` or `127.0.0.1` endpoints.
4. Record `local_only=true` and `cloud_api_calls=0`.
5. Save pilot outputs under a clearly labeled path such as
   `results/local_llm_pilot/`.
6. Keep pilot tables separate from `results/fair_comparison/`.
7. Add a report note that pilot results are not formal method evidence.

## 5. Suggested Team Message

Local LLM remains frozen as a pilot path. The default answerer is still
Template Answer. Please do not install Ollama, download models, change demo
defaults, or cite Local LLM outputs as formal results. Our formal conclusions
continue to come from retrieval-level fair comparison and resource experiments.

