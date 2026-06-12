"""Tests for the experiment-tracking utilities."""

from __future__ import annotations

import json
from pathlib import Path

from biomllab.tracking import Run, load_run, log_run


def test_log_run_writes_json_file(tmp_path: Path) -> None:
    path = log_run(
        "unit",
        params={"lr": 0.1, "seed": 0},
        metrics={"acc": 0.9},
        runs_dir=tmp_path,
    )
    assert path.exists()
    assert path.parent == tmp_path
    assert path.suffix == ".json"


def test_logged_run_roundtrips(tmp_path: Path) -> None:
    path = log_run("roundtrip", params={"k": 1}, metrics={"m": 2}, runs_dir=tmp_path)
    run = load_run(path)
    assert run.name == "roundtrip"
    assert run.params == {"k": 1}
    assert run.metrics == {"m": 2}
    assert run.run_id
    assert run.timestamp


def test_run_json_is_stable_and_sorted() -> None:
    run = Run(
        name="x",
        params={"b": 1, "a": 2},
        metrics={},
        run_id="fixed123",
        timestamp="2026-06-22T00:00:00+00:00",
        git_sha=None,
    )
    data = json.loads(run.to_json())
    assert data["run_id"] == "fixed123"
    # sort_keys=True => params serialize alphabetically
    assert list(data["params"].keys()) == ["a", "b"]


def test_runs_dir_is_created_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "runs"
    assert not target.exists()
    log_run("create", runs_dir=target)
    assert target.is_dir()


def test_distinct_runs_get_distinct_files(tmp_path: Path) -> None:
    p1 = log_run("a", runs_dir=tmp_path)
    p2 = log_run("b", runs_dir=tmp_path)
    assert p1 != p2
    assert len(list(tmp_path.glob("*.json"))) == 2
