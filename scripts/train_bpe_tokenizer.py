from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from datasets import load_dataset


def _iter_texts(ds, *, text_field: str) -> Iterable[str]:
    for ex in ds:
        text = ex.get(text_field)
        if isinstance(text, str) and text:
            yield text


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer on TinyStories.")
    parser.add_argument("--dataset", default="roneneldan/TinyStories", help="HuggingFace dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to use (e.g. train/validation).")
    parser.add_argument("--text-field", default="text", help="Field containing the raw text.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on number of examples.")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Tokenizer vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency.")
    parser.add_argument("--output", default="data/tinystories_bpe.json", help="Output tokenizer JSON path.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase during normalization.")
    args = parser.parse_args()

    from therapml.phase2.part1.tokenizer.bpe import BPETokenizer, BPETrainingConfig

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples is not None:
        ds = ds.select(range(int(args.max_examples)))

    config = BPETrainingConfig(
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
        lowercase=bool(args.lowercase),
    )

    tokenizer = BPETokenizer.train_from_iterator(_iter_texts(ds, text_field=args.text_field), config=config)
    out_path = tokenizer.save(Path(args.output))
    print(f"Saved tokenizer: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

