# bio-ml-lab

Professional Python baseline and lab repo for my path to **Research Scientist, Life Sciences**.
This is the **Phase 1 deliverable** of the 6-month plan: a reproducible, production-standard repo where the
Phase 1–2 experiments (PyTorch consolidation, fine-tuning, LoRA/QLoRA, SFT/DPO) will live.

> Week 1 exit criterion: on a clean machine, `uv pip install -e ".[dev]"` and `pytest` both run green. ✅
> Week 2 exit criterion: green CI on a PR + first tracked run; Docker image builds and tests pass inside it. ✅
> Week 4 exit criterion: a reproducible fine-tune with a registered baseline metric. ✅

## Quickstart

```bash
# 1. Create the virtual env and install the package + dev tools
uv venv
uv pip install -e ".[dev]"

# 2. Activate (optional; `uv run` works without activating)
source .venv/bin/activate

# 3. Run the full quality gate (what CI runs)
uv run ruff check .      # lint
uv run ruff format --check .
uv run mypy              # strict type-check
uv run pytest            # tests
```

Or use the Makefile shortcuts:

```bash
make install   # venv + editable install
make check     # lint + type + tests
```

## CLI + experiment tracking (Week 2)

```bash
# Compute sequence stats and record a tracked run under runs/
biomllab stats --name gc-baseline ATGCGC GGGGCC ATATAT
```

Each run is written as a self-contained JSON file with **params, metrics, a
timestamp and the git SHA** (provenance) — no account, no API key, no secrets.
See `runs/example-seq-stats.json` for a committed example. When a Phase 2
experiment needs richer dashboards, an optional W&B/DVC backend can sit behind
the same `log_run` call.

## Docker (Week 2)

```bash
make docker-build   # build the reproducible image
make docker-test    # run the test suite inside the container
```

CI builds the image and runs the tests inside it on every PR, so "works on my
machine" is not a thing here.

## Fine-tune (Week 4)

A small, fully reproducible fine-tune: classify whether a DNA sequence contains
a **TATA-box** promoter motif, fine-tuning `google/bert_uncased_L-2_H-128_A-2`
(BERT-mini) with the Hugging Face `Trainer` on CPU. Data is synthetic and
seed-fixed; each run is tracked via `log_run`.

```bash
uv pip install -e ".[ml]"                  # CPU-only torch + transformers
uv run biomllab finetune --epochs 8        # one tracked run
uv run python scripts/sweep_epochs.py      # learning curve -> results/epochs_sweep.csv
```

Accuracy climbs with training (majority baseline = 0.50):

| Epochs | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| Accuracy | 0.50 | 0.70 | 0.76 | 0.82 | **0.87** |

Full data in [`results/epochs_sweep.csv`](results/epochs_sweep.csv); details and
caveats in the [model card](src/biomllab/finetune/MODEL_CARD.md).

## Structure

```
bio-ml-lab/
├── pyproject.toml            # build + ruff + mypy + pytest config (single source of truth)
├── Makefile                  # install / lint / format / type / test / check / docker-*
├── Dockerfile                # reproducible image (uv base); CI builds + tests inside it
├── .pre-commit-config.yaml   # ruff + mypy + hygiene hooks
├── .github/workflows/ci.yml  # CI: matrix py3.10/3.12 + docker build/test
├── src/biomllab/             # the package (src layout)
│   ├── __init__.py
│   ├── sequences.py          # typed DNA utilities (toolchain smoke test)
│   ├── tracking.py           # dependency-free experiment/run tracking
│   ├── cli.py                # `biomllab` CLI (stats / finetune)
│   └── finetune/             # Week 4: reproducible fine-tune
│       ├── data.py           # synthetic, seed-fixed TATA-box dataset
│       ├── train.py          # HF Trainer pipeline + epochs sweep
│       └── MODEL_CARD.md
├── scripts/sweep_epochs.py   # runs the learning-curve sweep
├── results/epochs_sweep.csv  # committed sweep results (data for the curve)
├── runs/                     # tracked runs (one example committed; rest gitignored)
└── tests/
    ├── test_sequences.py
    ├── test_tracking.py
    ├── test_cli.py
    ├── test_finetune_data.py
    └── test_finetune_train.py   # skipped unless [ml] is installed
```

## Engineering standards (the point of this repo)

- **src layout** so tests import the installed package, not the working tree.
- **Typed everywhere**, `mypy --strict` must pass.
- **Lint + format** with ruff; **pre-commit** runs them before every commit.
- **Reproducible** env with `uv`; pinned dev tools.
- **Green CI** on every PR (this is the signal Anthropic looks for).

## Pre-commit (run once)

```bash
uv run pre-commit install
```

## Roadmap

- **Phase 1 (wk 1–4):** this baseline → consolidate Transformers → first reproducible fine-tune.
- **Phase 2 (wk 5–9):** LoRA/QLoRA → SFT → DPO experiments as subpackages here.
- **Phase 3 (wk 10–19):** the flagship `bio-research-agent-harness` lives in its own repo; this lab feeds it.

See `~/int_anthropic/01_plan_6_meses.md` for the full plan.
