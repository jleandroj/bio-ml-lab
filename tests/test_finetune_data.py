"""Tests for the synthetic DNA dataset generator."""

from __future__ import annotations

import pytest

from biomllab.finetune.data import (
    TATA_BOX,
    make_dataset,
    make_examples,
    to_kmers,
)


def test_to_kmers_basic() -> None:
    assert to_kmers("ATGCA", k=3) == "ATG TGC GCA"


def test_to_kmers_shorter_than_k_returns_sequence() -> None:
    assert to_kmers("AT", k=3) == "AT"


def test_to_kmers_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        to_kmers("ATGC", k=0)


def test_make_examples_is_balanced() -> None:
    examples = make_examples(100, seed=0)
    labels = [e.label for e in examples]
    assert sum(labels) == 50  # exactly half positive


def test_positive_examples_contain_motif_negatives_do_not() -> None:
    examples = make_examples(200, seed=1)
    for e in examples:
        if e.label == 1:
            assert TATA_BOX in e.sequence
        else:
            assert TATA_BOX not in e.sequence


def test_sequences_use_only_dna_bases() -> None:
    examples = make_examples(50, seed=2)
    for e in examples:
        assert set(e.sequence) <= set("ACGT")


def test_generation_is_reproducible() -> None:
    a = make_examples(30, seed=7)
    b = make_examples(30, seed=7)
    assert a == b


def test_different_seed_changes_data() -> None:
    a = make_examples(30, seed=7)
    b = make_examples(30, seed=8)
    assert a != b


def test_seq_len_must_fit_motif() -> None:
    with pytest.raises(ValueError):
        make_examples(10, seq_len=3)


def test_make_dataset_splits_do_not_overlap() -> None:
    ds = make_dataset(n_train=100, n_val=40, seed=0)
    assert len(ds.train) == 100
    assert len(ds.val) == 40
    train_seqs = {e.sequence for e in ds.train}
    val_seqs = {e.sequence for e in ds.val}
    assert train_seqs.isdisjoint(val_seqs)
