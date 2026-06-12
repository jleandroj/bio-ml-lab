"""Run an epochs sweep and save the results for a learning curve.

Fine-tunes the BERT-mini model on the TATA-box task for several epoch budgets
(seed fixed) and writes one row per run to ``results/epochs_sweep.csv``. That CSV
is committed as the data behind the accuracy-vs-epochs curve we plot later.

Usage:
    uv run python scripts/sweep_epochs.py
    uv run python scripts/sweep_epochs.py 1 2 3 5 8
"""

from __future__ import annotations

import sys
from pathlib import Path

from biomllab.finetune.train import run_sweep, write_results_csv

DEFAULT_EPOCHS = [1.0, 2.0, 3.0, 5.0, 8.0]
OUTPUT = Path("results/epochs_sweep.csv")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    epochs = [float(a) for a in args] if args else DEFAULT_EPOCHS
    print(f"Sweeping epochs={epochs} ...")
    results = run_sweep(epochs)
    path = write_results_csv(results, OUTPUT)
    print(f"\nWrote {len(results)} rows to {path}\n")
    for row in results:
        print(
            f"  epochs={row['epochs']:>4}  acc={row['eval_accuracy']:.3f}  "
            f"f1={row['eval_f1']:.3f}  baseline={row['majority_baseline_accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
