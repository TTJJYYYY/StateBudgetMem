# Development tools

These are developer-facing entry points, not importable project modules.

- `memorybank/`: full baseline evaluation and stale-memory analysis wrappers.
- `routing/`: prompt inspection and real-API router diagnostics.

Reusable logic must stay under `src/statebudgetmem/`; tools should remain thin
entry points around that logic.
