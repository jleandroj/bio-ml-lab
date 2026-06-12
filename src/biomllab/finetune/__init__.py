"""Fine-tuning subpackage: a small, reproducible DNA-classification fine-tune.

The task is deliberately simple but biologically motivated: decide whether a DNA
sequence contains a TATA-box motif (a core promoter element). Data is synthetic
and seed-fixed so the whole pipeline runs offline, on CPU, in minutes — the
point is a *reproducible fine-tune*, not a state-of-the-art genomics model.
"""

from __future__ import annotations

from biomllab.finetune.data import (
    TATA_BOX,
    Dataset,
    Example,
    make_dataset,
    make_examples,
    to_kmers,
)

__all__ = [
    "TATA_BOX",
    "Dataset",
    "Example",
    "make_dataset",
    "make_examples",
    "to_kmers",
]
