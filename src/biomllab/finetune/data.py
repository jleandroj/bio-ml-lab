"""Synthetic DNA dataset for the TATA-box classification fine-tune.

A *promoter* in molecular biology often contains a TATA-box: a short, conserved
motif (canonically ``TATAAA``) sitting upstream of the transcription start site.
Here we build a clean, balanced, fully reproducible toy task:

- label 1: the sequence contains the motif (a promoter-like sequence);
- label 0: a random sequence that does *not* contain the motif.

Sequences are converted to space-separated k-mers (the tokenization scheme used
by DNABERT-style models) so a WordPiece tokenizer can handle them.

Everything is driven by a fixed seed: same seed in, same data out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: The four DNA bases.
BASES: tuple[str, ...] = ("A", "C", "G", "T")

#: Canonical TATA-box motif used as the positive signal.
TATA_BOX: str = "TATAAA"


@dataclass(frozen=True)
class Example:
    """One labelled DNA sequence.

    Attributes:
        sequence: The raw DNA string (only A/C/G/T).
        label: 1 if the sequence contains the promoter motif, else 0.
    """

    sequence: str
    label: int


@dataclass(frozen=True)
class Dataset:
    """A train/validation split of :class:`Example` items."""

    train: list[Example]
    val: list[Example]


def to_kmers(sequence: str, k: int = 3) -> str:
    """Convert a DNA sequence to space-separated overlapping k-mers.

    >>> to_kmers("ATGCA", k=3)
    'ATG TGC GCA'
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(sequence) < k:
        return sequence
    return " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def _random_sequence(rng: random.Random, length: int) -> str:
    """Draw a uniform random DNA sequence of the given length."""
    return "".join(rng.choice(BASES) for _ in range(length))


def make_examples(
    n: int,
    *,
    seq_len: int = 60,
    motif: str = TATA_BOX,
    seed: int = 0,
) -> list[Example]:
    """Generate ``n`` balanced, labelled examples.

    Half the examples contain ``motif`` inserted at a random position (label 1);
    the other half are random sequences guaranteed not to contain it (label 0).
    The result is shuffled deterministically.

    Raises:
        ValueError: if ``seq_len`` is shorter than ``motif``.
    """
    if seq_len < len(motif):
        raise ValueError(f"seq_len ({seq_len}) must be >= len(motif) ({len(motif)})")
    rng = random.Random(seed)
    examples: list[Example] = []
    for i in range(n):
        label = i % 2  # balanced by construction
        if label == 1:
            seq = _random_sequence(rng, seq_len)
            pos = rng.randint(0, seq_len - len(motif))
            seq = seq[:pos] + motif + seq[pos + len(motif) :]
        else:
            seq = _random_sequence(rng, seq_len)
            while motif in seq:  # keep negatives clean
                seq = _random_sequence(rng, seq_len)
        examples.append(Example(sequence=seq, label=label))
    rng.shuffle(examples)
    return examples


def make_dataset(
    n_train: int = 800,
    n_val: int = 200,
    *,
    seq_len: int = 60,
    motif: str = TATA_BOX,
    seed: int = 0,
) -> Dataset:
    """Build a reproducible train/validation split.

    Train and validation use different seeds so they do not share sequences.
    """
    train = make_examples(n_train, seq_len=seq_len, motif=motif, seed=seed)
    val = make_examples(n_val, seq_len=seq_len, motif=motif, seed=seed + 1)
    return Dataset(train=train, val=val)
