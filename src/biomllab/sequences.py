"""Small, fully-typed DNA sequence utilities.

These exist to prove the toolchain works end-to-end (typing, tests, lint) with
a function from your own domain. They are intentionally simple but written to
production standard: validated input, clear errors, documented behaviour.
"""

from __future__ import annotations

_COMPLEMENT: dict[str, str] = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "N": "N",
}


def _validate(seq: str) -> str:
    """Uppercase ``seq`` and ensure every base is one of A, T, G, C, N.

    Raises:
        ValueError: if the sequence contains an unsupported character.
    """
    upper = seq.upper()
    invalid = sorted(set(upper) - _COMPLEMENT.keys())
    if invalid:
        raise ValueError(f"unsupported base(s) in sequence: {invalid}")
    return upper


def gc_content(seq: str) -> float:
    """Return the GC fraction of a DNA sequence (0.0-1.0).

    An empty sequence returns 0.0. ``N`` bases count toward the length but not
    toward the GC count.

    >>> gc_content("GGCC")
    1.0
    >>> gc_content("ATAT")
    0.0
    """
    if not seq:
        return 0.0
    upper = _validate(seq)
    gc = sum(1 for base in upper if base in ("G", "C"))
    return gc / len(upper)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    >>> reverse_complement("ATGC")
    'GCAT'
    """
    upper = _validate(seq)
    return "".join(_COMPLEMENT[base] for base in reversed(upper))
