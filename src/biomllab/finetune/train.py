"""Reproducible fine-tune of a small transformer on the TATA-box task.

This is the Week 4 deliverable: a *reproducible fine-tune* with a registered
baseline metric. We fine-tune ``google/bert_uncased_L-2_H-128_A-2`` (BERT-mini,
2 layers) with the Hugging Face ``Trainer``. DNA sequences are turned into
k-mers; we pair the pretrained weights with our own k-mer tokenizer and resize
the model's embeddings, so the pretrained transformer blocks are genuinely
fine-tuned on a domain they never saw.

Everything is seeded and the run is logged via :func:`biomllab.tracking.log_run`
(params + metrics + git SHA), so results are reproducible and comparable.

For offline tests, :func:`build_kmer_tokenizer` and :func:`build_tiny_model`
construct a tokenizer and a tiny model locally with no network access.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from biomllab.finetune.data import BASES, Dataset, make_dataset, to_kmers
from biomllab.tracking import log_run


@dataclass
class FineTuneConfig:
    """All knobs for a fine-tune run (everything reproducible from this)."""

    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    k: int = 3
    seq_len: int = 60
    n_train: int = 800
    n_val: int = 200
    epochs: float = 3.0
    batch_size: int = 16
    learning_rate: float = 5e-4
    seed: int = 0
    max_length: int = 64
    output_dir: str = "outputs/tata-bert-tiny"
    track: bool = True


def all_kmers(k: int) -> list[str]:
    """Return every possible k-mer over the DNA alphabet, sorted."""
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def build_kmer_tokenizer(k: int) -> Any:
    """Build a fast WordLevel tokenizer whose vocabulary is the DNA k-mers.

    k-mers are whole tokens separated by spaces, so a WordLevel model with a
    whitespace pre-tokenizer is the natural fit (no sub-word splitting). Built
    entirely in-memory: no network, no files.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {tok: i for i, tok in enumerate(specials + all_kmers(k))}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def build_tiny_model(tokenizer: Any, *, num_labels: int = 2) -> Any:
    """Build a tiny ``BertForSequenceClassification`` from scratch (no download).

    Used for fast, offline smoke tests of the training loop.
    """
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(  # type: ignore[call-arg, unused-ignore]
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


def load_pretrained(model_name: str, k: int) -> tuple[Any, Any]:
    """Load pretrained weights and pair them with our DNA k-mer tokenizer.

    We deliberately use our own k-mer tokenizer (not the model's English
    WordPiece one, which is a poor fit for DNA) and resize the model's embedding
    table to the k-mer vocabulary. The pretrained transformer blocks (attention
    + feed-forward) are carried over and genuinely fine-tuned; the embeddings
    adapt to the new vocabulary during training.
    """
    from transformers import AutoModelForSequenceClassification

    tokenizer = build_kmer_tokenizer(k)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(tokenizer.vocab_size)
    return model, tokenizer


class _EncodedDataset:
    """A minimal torch-style dataset of pre-tokenized examples."""

    def __init__(self, encodings: dict[str, Any], labels: list[int]) -> None:
        import torch

        self._input_ids = encodings["input_ids"]
        self._attention_mask = encodings["attention_mask"]
        self._labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_mask[idx],
            "labels": self._labels[idx],
        }


def _encode(examples: Any, tokenizer: Any, k: int, max_length: int) -> _EncodedDataset:
    texts = [to_kmers(e.sequence, k=k) for e in examples]
    labels = [e.label for e in examples]
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return _EncodedDataset(enc, labels)


def majority_baseline_accuracy(dataset: Dataset) -> float:
    """Accuracy of always predicting the most common training label."""
    train_labels = [e.label for e in dataset.train]
    majority = 1 if sum(train_labels) * 2 >= len(train_labels) else 0
    correct = sum(1 for e in dataset.val if e.label == majority)
    return correct / len(dataset.val)


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }


def train_and_evaluate(
    config: FineTuneConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run the fine-tune and return a metrics dict.

    If ``model``/``tokenizer`` are provided they are used as-is (offline tests);
    otherwise the pretrained model named in ``config`` is downloaded.
    """
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(config.seed)
    dataset = make_dataset(config.n_train, config.n_val, seq_len=config.seq_len, seed=config.seed)

    if model is None or tokenizer is None:
        model, tokenizer = load_pretrained(config.model_name, config.k)

    train_ds = _encode(dataset.train, tokenizer, config.k, config.max_length)
    val_ds = _encode(dataset.val, tokenizer, config.k, config.max_length)

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    baseline = majority_baseline_accuracy(dataset)
    accuracy = float(eval_metrics.get("eval_accuracy", 0.0))
    metrics: dict[str, Any] = {
        "eval_accuracy": accuracy,
        "eval_f1": float(eval_metrics.get("eval_f1", 0.0)),
        "majority_baseline_accuracy": baseline,
        "improvement_over_baseline": accuracy - baseline,
    }

    if config.track:
        log_run(
            name=f"finetune-tata-{config.model_name.split('/')[-1]}",
            params=asdict(config),
            metrics=metrics,
        )
    return metrics


#: Columns written to the sweep CSV, in order.
SWEEP_FIELDS: tuple[str, ...] = (
    "epochs",
    "seed",
    "model_name",
    "eval_accuracy",
    "eval_f1",
    "majority_baseline_accuracy",
    "improvement_over_baseline",
)


def run_sweep(
    epochs_values: list[float],
    *,
    base_config: FineTuneConfig | None = None,
) -> list[dict[str, Any]]:
    """Fine-tune once per value in ``epochs_values`` and collect the results.

    Each run reuses ``base_config`` (seed fixed) with only ``epochs`` changed, so
    the resulting points form a clean, reproducible learning curve.
    """
    base = base_config or FineTuneConfig()
    results: list[dict[str, Any]] = []
    for epochs in epochs_values:
        config = replace(base, epochs=float(epochs))
        metrics = train_and_evaluate(config)
        results.append(
            {
                "epochs": float(epochs),
                "seed": config.seed,
                "model_name": config.model_name,
                **metrics,
            }
        )
    return results


def write_results_csv(results: list[dict[str, Any]], path: Path) -> Path:
    """Write sweep results to a CSV (created/overwritten) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SWEEP_FIELDS))
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in SWEEP_FIELDS})
    return path
