# Phase 1: On-device MemoryBank Core Baseline Results

## Scope and conclusion

Phase 1 now reaches a **credible, runnable, evaluable on-device MemoryBank core baseline** for the deliberately limited memory-system scope. Local structured memories are embedded by a cached MiniLM model, indexed and retrieved with local FAISS, reinforced after recall, evaluated with deterministic timestamps, persisted to local files, and measured without cloud APIs or a cloud database.

It is not a complete conversational-agent baseline. Memory extraction, automatic summary/profile evolution and answer generation are outside this phase and do not yet have a local LLM backend.

## Environment and commands

- OS: Windows laptop (full string in `results/ondevice_memorybank/baseline_run/environment.json`)
- Python: 3.13.5
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`, cached before the run and loaded with local-only mode
- Index: FAISS `IndexFlatIP`
- Cloud/API calls during experiment: 0
- Formal grid: 100/500/1000/2000 memories, Top-K 1/3/5, repeat 3, seed 42

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .tmp\pytest-full -p no:cacheprovider

.venv\Scripts\python.exe tools\memorybank\run_ondevice_memorybank_baseline.py `
  --output-dir results\ondevice_memorybank\baseline_run `
  --embedding-backend sentence-transformers `
  --memory-counts 100 500 1000 2000 --top-k 1 3 5 --repeat 3 `
  --seed 42 --local-only --detailed-logs `
  --enable-forgetting --enable-reinforcement
```

## Test result

`290 passed in 11.53s`. The initial run used the inaccessible system temp directory and produced 10 setup errors; rerunning with a workspace `--basetemp` produced no failures.

## Key results

The dataset contains fixed controlled current/stale facts and deterministic synthetic neutral distractors. It is a resource-scaling experiment, not evidence of real-user accuracy.

| Memories | Recall@1 | Valid Recall@1 | Stale rate@5 | Mean latency@5 (ms) | P95 latency@5 (ms) | Write+embed (ms) | FAISS bytes | Total core storage bytes | Build peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.000 | 1.000 | 0.000 | 18.280 | 20.546 | 2,002 | 153,645 | 369,508 | 491,513,173 |
| 500 | 1.000 | 1.000 | 0.000 | 18.070 | 19.605 | 9,617 | 768,045 | 1,848,448 | 496,885,760 |
| 1,000 | 1.000 | 1.000 | 0.000 | 18.067 | 20.368 | 18,993 | 1,536,045 | 3,697,128 | 504,629,931 |
| 2,000 | 1.000 | 1.000 | 0.000 | 17.492 | 20.065 | 38,290 | 3,072,045 | 7,394,472 | 517,768,533 |

Prompt token estimate at Top-K 1/3/5 was 8.25/24.25/44.75 on average. It is explicitly a whitespace-word plus non-ASCII-character proxy, not a model tokenizer count.

The 100% controlled recall reflects easy, deliberately unambiguous probes and strong MiniLM semantics. It must not be compared directly with the MemoryBank paper's human-scored 194-question experiment.

## Mechanism evidence

- New `MemoryPiece.strength` starts at 1.
- Retrieval logs contain `before_strength`, `after_strength`, before/after access times, rank, score and retention.
- Selected memories receive `strength += 1` and `last_accessed = current_time`.
- Forgetting logs deterministically compute `R=exp(-t/S)` and list `forgotten_memory_ids`.
- The runner only evaluates threshold-based forgotten candidates. It does not apply the project's non-paper `strength *= 0.5` mutation.
- Reload tests and the formal run load the persisted FAISS index before evaluation.

## Artifacts

Formal output: `results/ondevice_memorybank/baseline_run/`

- `config.json`, `environment.json`
- `predictions.csv`, `metrics.json`
- `resource_metrics.csv`, `memorybank_resource_metrics.json`
- `memorybank_retrieval_log.jsonl`
- `memorybank_reinforcement_log.jsonl`
- `memorybank_forgetting_log.jsonl`
- `memorybank_run_summary.json`, `summary.md`
- `storage/n*_r*/memory_data.jsonl`
- `storage/n*_r*/metadata.json`
- `storage/n*_r*/embeddings.npy`
- `storage/n*_r*/memorybank.faiss`
- seven PNGs under `figures/`

## Findings and limitations

The current store path embeds each memory separately. Write+embedding time therefore grows almost linearly and reaches about 38.3 seconds at 2,000 memories; a later fair optimization should batch embeddings without changing the baseline retrieval rule. Detailed forgetting logs also scale with every memory and repeat: the formal JSONL logs occupy about 322 MB. This is useful evidence of a logging/storage pressure point, but production experiments should offer compact event-only logs while retaining aggregate counts.

RSS includes the Python process, PyTorch and MiniLM model, not only the FAISS index. This is intentional for laptop process cost, while the separately reported FAISS bytes isolate index storage. `index_build_ms` is zero because the current implementation updates `IndexFlatIP` incrementally during each store; the combined `embedding_and_index_build_ms`/`embedding_ms` field captures that path. A future measurement should separate batched encode and a single FAISS add operation.

The next phase can connect TF-IDF, native MemoryBank, MemoryBank+Versioning, +Views, +Routing and full StateBudgetMem to this runner's dataset/config/output contracts. It must keep the same memory inputs, encoder, seed, Top-K, token/storage budgets and hardware, then add valid recall, stale retrieval/use and answer metrics without changing this baseline's paper-compatible reinforcement semantics.
