<<<<<<< HEAD
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
=======
# 基础记忆系统与基线

本目录负责项目中的基础对照方法。

主要内容：

- 现有 TF-IDF + Cosine Similarity 基线；
- MemoryBank 复现与整理；
- 基础向量检索；
- 时间衰减、重要性和记忆强化；
- 与其他方法使用统一数据和指标进行比较。
>>>>>>> ba900d42c9450c7df9e9737f2bedadadbdce7427
