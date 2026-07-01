# Baselines

Reference methods are organized by method rather than scattered across app,
evaluation, and script folders.

```text
baselines/
├── memorybank/   # system, agents, datasets, evaluator, staleness, demo
└── tfidf/        # retriever, adapter, controlled experiment runner
```

Baseline-specific code stays inside its own package. Only contracts and metrics
that are genuinely shared by multiple methods belong in `core/`, `retrieval/`,
or `evaluation/`.
