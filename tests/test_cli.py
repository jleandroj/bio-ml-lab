"""Tests for the biomllab command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomllab.cli import _stats, main


def test_stats_empty() -> None:
    assert _stats([]) == {"n_sequences": 0, "mean_gc": 0.0, "total_bases": 0}


def test_stats_computes_mean_gc() -> None:
    # GGCC -> 1.0, ATAT -> 0.0  => mean 0.5
    result = _stats(["GGCC", "ATAT"])
    assert result["n_sequences"] == 2
    assert result["mean_gc"] == pytest.approx(0.5)
    assert result["min_gc"] == pytest.approx(0.0)
    assert result["max_gc"] == pytest.approx(1.0)
    assert result["total_bases"] == 8


def test_main_stats_writes_tracked_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    exit_code = main(["stats", "--runs-dir", str(runs_dir), "GGCC", "ATAT"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "mean GC:" in out
    assert "tracked ->" in out
    assert len(list(runs_dir.glob("*.json"))) == 1


def test_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
