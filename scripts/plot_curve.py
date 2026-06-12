"""Plot the accuracy-vs-epochs learning curve from the sweep CSV.

Reads ``results/epochs_sweep.csv`` (produced by ``scripts/sweep_epochs.py``) and
writes ``results/epochs_curve.png``. Pure stdlib + matplotlib; no pandas.

Usage:
    uv run python scripts/plot_curve.py        # requires the [viz] extra
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = Path("results/epochs_sweep.csv")
OUT_PATH = Path("results/epochs_curve.png")


def main() -> int:
    with CSV_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No data in {CSV_PATH}")

    epochs = [float(r["epochs"]) for r in rows]
    accuracy = [float(r["eval_accuracy"]) for r in rows]
    f1 = [float(r["eval_f1"]) for r in rows]
    baseline = float(rows[0]["majority_baseline_accuracy"])
    model = rows[0]["model_name"].split("/")[-1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, accuracy, "o-", label="accuracy", linewidth=2)
    ax.plot(epochs, f1, "s--", label="F1", linewidth=2)
    ax.axhline(baseline, color="gray", linestyle=":", label=f"majority baseline ({baseline:.2f})")

    ax.set_xlabel("training epochs")
    ax.set_ylabel("validation score")
    ax.set_title(f"TATA-box fine-tune learning curve\n({model}, seed 0, CPU)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
