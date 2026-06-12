"""Tests for the fine-tune pipeline.

These require the optional [ml] stack (torch + transformers + sklearn); when it
is not installed (e.g. the base CI job) they are skipped. They never hit the
network: the smoke test builds a tiny model and a k-mer tokenizer locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from biomllab.finetune.data import make_dataset  # noqa: E402
from biomllab.finetune.train import (  # noqa: E402
    SWEEP_FIELDS,
    FineTuneConfig,
    all_kmers,
    build_kmer_tokenizer,
    build_tiny_model,
    majority_baseline_accuracy,
    train_and_evaluate,
    write_results_csv,
)


def test_all_kmers_count() -> None:
    assert len(all_kmers(3)) == 64  # 4 ** 3
    assert all_kmers(1) == ["A", "C", "G", "T"]


def test_write_results_csv_roundtrip(tmp_path: Path) -> None:
    import csv

    rows = [
        {
            "epochs": 1.0,
            "seed": 0,
            "model_name": "m",
            "eval_accuracy": 0.5,
            "eval_f1": 0.4,
            "majority_baseline_accuracy": 0.5,
            "improvement_over_baseline": 0.0,
        },
        {
            "epochs": 3.0,
            "seed": 0,
            "model_name": "m",
            "eval_accuracy": 0.8,
            "eval_f1": 0.84,
            "majority_baseline_accuracy": 0.5,
            "improvement_over_baseline": 0.3,
        },
    ]
    out = write_results_csv(rows, tmp_path / "sweep.csv")
    with out.open(encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert list(read[0].keys()) == list(SWEEP_FIELDS)
    assert len(read) == 2
    assert read[1]["eval_accuracy"] == "0.8"


def test_kmer_tokenizer_knows_kmers() -> None:
    tok = build_kmer_tokenizer(k=3)
    ids = tok.convert_tokens_to_ids(["ATG", "TGC"])
    unk = tok.convert_tokens_to_ids(tok.unk_token)
    assert all(i != unk for i in ids)


def test_majority_baseline_is_reasonable() -> None:
    ds = make_dataset(n_train=100, n_val=40, seed=0)
    acc = majority_baseline_accuracy(ds)
    # Balanced data => majority baseline is around chance.
    assert 0.0 <= acc <= 1.0


def test_training_smoke_runs_and_tracks(tmp_path: Path) -> None:
    tok = build_kmer_tokenizer(k=3)
    model = build_tiny_model(tok)
    config = FineTuneConfig(
        k=3,
        n_train=32,
        n_val=16,
        epochs=1.0,
        batch_size=8,
        output_dir=str(tmp_path),
        track=False,  # don't pollute runs/ from tests
    )
    metrics = train_and_evaluate(config, model=model, tokenizer=tok)
    assert set(metrics) >= {
        "eval_accuracy",
        "eval_f1",
        "majority_baseline_accuracy",
        "improvement_over_baseline",
    }
    assert 0.0 <= metrics["eval_accuracy"] <= 1.0
