"""Run the controlled-data TF-IDF baseline from Python."""

from pathlib import Path

from statebudgetmem.baselines.tfidf import BaselineConfig, run_baseline


config = BaselineConfig(
    method="tfidf_topk",
    dataset_path=Path("data/controlled/baseline_scenarios.jsonl"),
    top_k=3,
    random_seed=42,
    results_dir=Path("results"),
    config_path=Path("configs/baseline.yaml"),
)
print(run_baseline(config))
