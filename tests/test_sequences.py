"""Tests for biomllab.sequences. Models the testing standard for the whole repo."""

from __future__ import annotations

import pytest

from biomllab import gc_content, reverse_complement


def test_gc_content_basic() -> None:
    assert gc_content("GGCC") == 1.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("ATGC") == 0.5


def test_gc_content_empty_is_zero() -> None:
    assert gc_content("") == 0.0


def test_gc_content_is_case_insensitive() -> None:
    assert gc_content("ggcc") == gc_content("GGCC")


def test_gc_content_counts_n_in_length_only() -> None:
    # 1 G out of 4 bases (N counts toward length, not GC)
    assert gc_content("GNNN") == pytest.approx(0.25)


def test_reverse_complement_basic() -> None:
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AAAA") == "TTTT"


def test_reverse_complement_is_involution() -> None:
    seq = "ATGCGTACGN"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_invalid_base_raises() -> None:
    with pytest.raises(ValueError, match="unsupported base"):
        gc_content("ATBX")
    with pytest.raises(ValueError, match="unsupported base"):
        reverse_complement("123")
