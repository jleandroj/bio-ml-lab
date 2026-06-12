# bio-ml-lab

Professional Python baseline and lab repo for my path to **Research Scientist, Life Sciences @ Anthropic**.
This is the **Week 1 deliverable** of the 6-month plan: a reproducible, production-standard repo where the
Phase 1–2 experiments (PyTorch consolidation, fine-tuning, LoRA/QLoRA, SFT/DPO) will live.

> Exit criterion for Week 1: on a clean machine, `uv pip install -e ".[dev]"` and `pytest` both run green.

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

## Structure

```
bio-ml-lab/
├── pyproject.toml            # build + ruff + mypy + pytest config (single source of truth)
├── Makefile                  # install / lint / format / type / test / check
├── .pre-commit-config.yaml   # ruff + mypy + hygiene hooks
├── .github/workflows/ci.yml  # CI: matrix py3.10/3.12, lint + type + test
├── src/biomllab/             # the package (src layout)
│   ├── __init__.py
│   └── sequences.py          # typed DNA utilities (toolchain smoke test)
└── tests/
    └── test_sequences.py
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
