"""bio-ml-lab: professional Python baseline for the Anthropic Life Sciences prep.

Phase 1-2 experiments (PyTorch consolidation, fine-tuning, LoRA, DPO) live in
subpackages added over time. This module exposes the package version and the
small sequence utilities used to validate the toolchain.
"""

from biomllab.sequences import gc_content, reverse_complement

__version__ = "0.1.0"

__all__ = ["__version__", "gc_content", "reverse_complement"]
