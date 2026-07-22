# StateBudgetMem Final Showcase

Open `index.html` directly in a browser. The page is self-contained and does
not require a local server, a cloud API, or a live LLM by default.

## Layers

1. Case Entry: a short fixed dialogue used only as a presentation entry.
2. MemoryExplorer: fixed no-free-input cases for temporary invalidation,
   permanent supersede, and historical/change queries.
3. Free Question Demo: browser-side three-column comparison over the same fixed
   demo memories. This is illustrative only and is not used for formal metrics.
4. Experiment Dashboard: formal fair-comparison metrics loaded from
   `D:\projects\memorybankproject\StateBudgetMem\results\fair_comparison_v2`.

## Optional Local LLM

- Requested answerer: `template`
- Actual answerer: `template`
- Model: `deterministic_template_v1`
- Local only: `True`
- Cloud API calls: `0`

To try Ollama locally, run:

```powershell
.venv\Scripts\python.exe tools\demo\build_final_showcase.py --answerer local_llm --local-llm-model <model_name>
```

If Ollama or the requested model is unavailable, the page is still generated
with the Template Answer and records the fallback reason in `showcase_data.json`.

## Optional DeepSeek Free-question Answers

The Free Question Demo can optionally ask DeepSeek to organize the three-column
answers. This is demo-only and still not part of formal metrics.

```powershell
.venv\Scripts\python.exe tools\demo\run_final_showcase_server.py
```

Then open `http://127.0.0.1:8765/index.html`, select `DeepSeek API via local
demo server`, type the DeepSeek API key into the page, and run the comparison.
The API key is sent to the local demo server for that request only; it is not
written into HTML, JSON, logs, or result files.

## Formal Experiment Source

- Dataset: `data\controlled\temporal_challenge_v1.jsonl`
- Config: `configs\fair_comparison_memorybank_statebudgetmem_v2.yaml`
- Run ID: `unified_smoke_seed42_20260718T071439266732Z`
- Top-K: `3`
- Candidate-K: `20`
- Token budget: `256`
- Seed: `42`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`

## Boundary

The Case Entry, MemoryExplorer, and Free Question Demo are display and analysis
tools. The free-question area uses browser-side template retrieval/answering
only; it does not call Ollama, does not extract new memories, and does not
update memory state. Formal performance conclusions come from the unified
runner outputs in `results/fair_comparison_v2/`.
