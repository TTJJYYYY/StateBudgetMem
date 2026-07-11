# On-device MemoryBank Core Baseline

- Status: success
- Local-only: true
- Cloud/API calls: 0
- Scope: memory-system baseline; no local LLM answer generation

The run uses local files, local embeddings and FAISS. Threshold-based forgotten candidates are
reported as a project evaluation policy; this runner does not apply the non-paper `strength *= 0.5` extension.
