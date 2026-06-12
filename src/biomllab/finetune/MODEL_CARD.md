# Model card — TATA-box classifier (BERT-mini fine-tune)

> Week 4 deliverable of the bio-ml-lab plan: a small, fully reproducible
> fine-tune. The goal is to demonstrate the *full reproducible loop* (data →
> tokenization → fine-tune → eval → tracking → curve), not state-of-the-art
> genomics.

## Summary

| Field | Value |
|---|---|
| Base model | `google/bert_uncased_L-2_H-128_A-2` (BERT-mini, 2 layers, hidden 128) |
| Task | Binary classification: does a DNA sequence contain a TATA-box (`TATAAA`)? |
| Domain motivation | The TATA-box is a core promoter element upstream of many transcription start sites |
| Tokenization | DNA → overlapping 3-mers → WordLevel tokenizer (DNABERT-style) |
| Training | Hugging Face `Trainer`, CPU only, seed-fixed |
| Tracking | Each run logged via `biomllab.tracking.log_run` (params + metrics + git SHA) |

## Data

Synthetic and seed-fixed (`biomllab.finetune.data`):

- **Positive (label 1):** random DNA with `TATAAA` inserted at a random position.
- **Negative (label 0):** random DNA guaranteed *not* to contain the motif.
- Balanced 50/50. Default 800 train / 200 val, `seq_len=60`. Train and val use
  different seeds, so they share no sequences.

Synthetic data keeps the whole pipeline offline, reproducible, and frugal on
disk — at the cost of realism (see *Limitations*).

## Results

Accuracy vs training epochs (seed 0, majority baseline = 0.50). Data:
[`results/epochs_sweep.csv`](../../../results/epochs_sweep.csv).

| Epochs | Accuracy | F1 | Δ vs baseline |
|---:|---:|---:|---:|
| 1 | 0.500 | 0.000 | +0.00 |
| 2 | 0.695 | 0.601 | +0.20 |
| 3 | 0.760 | 0.791 | +0.26 |
| 5 | 0.820 | 0.841 | +0.32 |
| 8 | 0.870 | 0.881 | +0.37 |

The model learns the motif: accuracy rises monotonically and clearly beats the
majority-class baseline.

## How to reproduce

```bash
uv pip install -e ".[ml]"        # CPU-only torch
uv run biomllab finetune --epochs 8        # single run (tracked in runs/)
uv run python scripts/sweep_epochs.py      # full curve -> results/epochs_sweep.csv
```

Determinism: `transformers.set_seed(seed)` plus seeded data generation. Same
seed in → same metrics out (CPU).

## Limitations and honest caveats

- **Synthetic task.** A single fixed motif is far easier than real promoter
  prediction (variable motifs, context, chromatin, etc.). Numbers here do **not**
  transfer to real genomics.
- **Tiny model, CPU.** Chosen for reproducibility on a laptop, not performance.
- **Embeddings re-initialized** for the DNA k-mer vocabulary; only the
  transformer blocks are inherited from the pretrained checkpoint.
- No test-set beyond a single val split; no cross-validation.

## Intended use

Educational / portfolio evidence that the author can build a reproducible,
tracked fine-tuning pipeline with HF Trainer. Not for any real biological
decision-making.
