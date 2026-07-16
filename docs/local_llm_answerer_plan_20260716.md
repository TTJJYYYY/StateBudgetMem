# Local LLM Answerer Plan - 2026-07-16

## 1. Why add a local LLM answerer

StateBudgetMem now has retrieval-level fair comparison, prompt augmentation,
and a final local showcase. The next useful demo step is to show how retrieved
temporal memories can support an actual answer on the endpoint, without sending
personal memory to a cloud API.

The local LLM path is for:

- endpoint demo quality;
- answer-level pilot inspection;
- future answer accuracy and stale-usage evaluation design.

It is not used as formal evidence for the current fair comparison.

## 2. Role of Template Answer

`TemplateAnswerer` remains the default answer path. It is deterministic,
offline, fast, and stable for tests and meeting demos. It also preserves the
current boundary of the formal retrieval experiments: Recall@K, Valid Recall@K,
Stale Retrieval Rate, retrieval latency, and token proxy are still evaluated
without model-generation noise.

## 3. Local LLM connection

`LocalLLMAnswerer` currently targets Ollama only:

```text
http://localhost:11434/api/generate
```

The adapter:

- accepts only `localhost` or `127.0.0.1` HTTP endpoints;
- sends `stream=false`;
- uses temperature `0.0` for stable demos;
- supports timeout and model-name configuration;
- records local-only metadata and `cloud_api_calls=0`;
- falls back to the template answer in demo scripts when Ollama or the model is
  unavailable.

No cloud API is called, and no model is downloaded by the code.

## 4. How to run

Default template path:

```powershell
.venv\Scripts\python.exe tools\demo\run_minimal_memorybank_dialog_demo.py --answerer template
.venv\Scripts\python.exe tools\demo\build_final_showcase.py
```

Optional local LLM path, after the user has already installed Ollama and pulled
a model locally:

```powershell
.venv\Scripts\python.exe tools\demo\run_minimal_memorybank_dialog_demo.py --answerer local_llm --local-llm-model <model_name>
.venv\Scripts\python.exe tools\demo\build_final_showcase.py --answerer local_llm --local-llm-model <model_name>
```

If Ollama is not running or the model is missing, the scripts still complete
with Template Answer and record the fallback reason.

## 5. New on-device metrics

The answer result records:

- `answerer_type`;
- `model_name`;
- `generation_latency_ms`;
- `prompt_tokens` or prompt token proxy;
- `generated_tokens` or generated token proxy;
- `tokens_per_second`;
- `used_memory_ids`;
- `local_only=true`;
- `cloud_api_calls=0`.

The first version uses Ollama token counts when available. If exact tokenizer
counts are not available, it records deterministic token proxies while leaving
the result fields ready for a precise tokenizer later.

## 6. Relationship to fair comparison

Formal fair comparison remains retrieval-level:

- Recall@K;
- Valid Recall@K;
- Stale Retrieval Rate;
- retrieval latency;
- token proxy.

Local LLM results are demo and answer-level pilot artifacts only. They must not
be mixed into existing formal fair-comparison conclusions.

## 7. Limits and next steps

Current limits:

- Ollama is the only implemented local generation backend;
- no automatic model install or download is attempted;
- no exact tokenizer integration is included yet;
- answer-level metrics are not yet part of the unified formal runner;
- local LLM output quality depends on the user-provided local model.

Next steps:

- add optional exact tokenizer support for selected local models;
- define answer-level pilot fixtures and stale-usage labels;
- add an explicit answer-level runner separate from retrieval fair comparison;
- compare Template Answer and Local LLM Answer only under a clearly labeled
  pilot protocol.
