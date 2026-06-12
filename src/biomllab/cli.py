"""Command-line entry point for biomllab.

Provides a small, real command that computes summary statistics over DNA
sequences and records the result as a tracked run. This is the Week 2 "first
tracked run": a reproducible computation whose params and metrics land in
``runs/`` with provenance.

Usage:
    biomllab stats ATGCATGC GGGCCC
    biomllab stats --name gc-baseline ATGC
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from biomllab.sequences import gc_content, reverse_complement
from biomllab.tracking import DEFAULT_RUNS_DIR, log_run


def _stats(sequences: list[str]) -> dict[str, float]:
    """Compute aggregate metrics over a list of DNA sequences."""
    if not sequences:
        return {"n_sequences": 0, "mean_gc": 0.0, "total_bases": 0}
    gcs = [gc_content(s) for s in sequences]
    return {
        "n_sequences": len(sequences),
        "mean_gc": sum(gcs) / len(gcs),
        "min_gc": min(gcs),
        "max_gc": max(gcs),
        "total_bases": sum(len(s) for s in sequences),
    }


def _cmd_stats(args: argparse.Namespace) -> int:
    metrics = _stats(args.sequences)
    path = log_run(
        args.name,
        params={"sequences": args.sequences},
        metrics=metrics,
        runs_dir=Path(args.runs_dir),
    )
    print(f"sequences:  {metrics['n_sequences']}")
    print(f"mean GC:    {metrics['mean_gc']:.3f}")
    print(f"total bp:   {metrics['total_bases']}")
    if args.sequences:
        print(f"first rc:   {reverse_complement(args.sequences[0])}")
    print(f"tracked ->  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``biomllab`` CLI."""
    parser = argparse.ArgumentParser(
        prog="biomllab",
        description="Small bio utilities with built-in run tracking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="Summarize DNA sequences and track the run.")
    stats.add_argument("sequences", nargs="*", help="DNA sequences (A/T/G/C/N).")
    stats.add_argument(
        "--name", default="seq-stats", help="Run name recorded in the tracking file."
    )
    stats.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory where tracked runs are written.",
    )
    stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
