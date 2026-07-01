# Team workflow

The team may work directly on `main`, but only when the following rules are
followed consistently.

## 1. Suggested ownership

| Area | Primary responsibility |
|---|---|
| `baselines/memorybank/`, `tools/memorybank/` | MemoryBank baseline, demo, Memora evaluation |
| `baselines/tfidf/`, `evaluation/`, controlled experiments | deterministic baseline and metrics |
| `versioning/`, `preprocessing/` | state updates and structured-memory ingestion |
| `routing/`, `views/`, future pipeline/app | query routing, dual views, integration |

Ownership means “coordinate changes here,” not “others are forbidden to read or
help.”

## 2. Shared files require coordination

Announce changes before editing:

```text
src/statebudgetmem/interfaces.py
src/statebudgetmem/core/
src/statebudgetmem/schemas/
src/statebudgetmem/cli.py
pyproject.toml
README.md
```

Do not create a new `MemoryRecord`, `QueryType`, `ViewType`, or equivalent type
inside a feature folder.

## 3. Direct-main routine

Before work:

```bash
git switch main
git pull --rebase origin main
```

Before pushing:

```bash
pytest -q
git status
git add <only files related to this task>
git commit -m "feat: concise description"
git pull --rebase origin main
git push origin main
```

Keep commits small. Avoid one commit that mixes refactoring, new algorithms,
data changes, and documentation.

## 4. Test placement

Put tests in the mirrored folder. Examples:

```text
routing/router.py             → tests/routing/test_router.py
versioning/engine.py          → tests/versioning/test_engine.py
baselines/tfidf/adapter.py    → tests/baselines/tfidf/test_adapter.py
```

Use `tests/integration/` only when a test crosses module boundaries.

## 5. Definition of done

A change is complete only when:

1. public imports remain valid;
2. offline tests pass without an API key;
3. optional dependencies fail with a clear installation message;
4. data and result formats are documented;
5. the contributor can explain how the change supports the research question.
