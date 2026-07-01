# MemoryBank baseline package

All code specific to the MemoryBank reproduction and comparison is colocated
here. New MemoryBank-only functionality should be added here, with mirrored
tests under `tests/baselines/memorybank/`.

- `system.py`: memory backends
- `agents.py`: generation wrappers
- `datasets.py`: demo and external adapters
- `evaluator.py`: answer-level evaluation
- `staleness.py`: stale-memory analysis
- `demo.py`: Gradio visualization

Shared metrics must remain in `statebudgetmem/evaluation/`; final integrated
StateBudgetMem applications belong in `statebudgetmem/apps/`.
