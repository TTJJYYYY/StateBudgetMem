# Deterministic TF-IDF baseline

This package contains the complete offline controlled-data baseline:

- `retriever.py`: deterministic tokenizer, TF-IDF, cosine ranking
- `adapter.py`: shared `MemoryMethod` interface adapter
- `runner.py`: configuration, experiment execution, and result export

Controlled data stays in `data/controlled/` because it is shared by TF-IDF,
versioning, views, and the final StateBudgetMem method.
