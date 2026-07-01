# Memora dataset location

The three downloaded branches contained only an empty `Memora/` directory; the
dataset itself was not committed. Clone or download Memora separately and point
the evaluator at its `data` directory:

```bash
git clone https://github.com/geniesinc/Memora.git data/external/memora/repo
python scripts/run_memorybank_evaluation.py \
  --memora-dir data/external/memora/repo/data \
  --persona software_engineer \
  --period weekly
```

Expected layout:

```text
Memora/data/
├── weekly/<persona>/conversations/session_*.json
├── weekly/<persona>/evaluation_questions_<persona>.json
├── monthly/
└── quarterly/
```
