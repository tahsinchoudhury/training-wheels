from __future__ import annotations

import argparse
from pathlib import Path

from therapml.phase2.part1.tokenizer.bpe import BPETokenizer


def _default_texts() -> list[str]:
    return [
        "Once upon a time, there was a small dragon.",
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's not butter!",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a saved BPE tokenizer and run encode/decode examples.")
    parser.add_argument(
        "--tokenizer",
        default="data/tinystories_bpe.json",
        help="Path to a saved tokenizer JSON file.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to encode/decode (repeatable). If omitted, uses a few built-in examples.",
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="Keep special tokens in decode output (default: skip special tokens).",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}\n"
            "Train it first with: `python scripts/train_bpe_tokenizer.py`"
        )

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    print("Note: ByteLevel BPE tokenizers often round-trip with a leading space (add_prefix_space behavior).")

    texts = list(args.text) or _default_texts()
    skip_special_tokens = not bool(args.keep_special_tokens)

    # print("\nSingle examples:")
    # for i, text in enumerate(texts, start=1):
    #     ids = tokenizer.encode(text)
    #     decoded = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    #     print(f"\n[{i}] text:    {text!r}")
    #     print(f"[{i}] ids:     {ids}")
    #     print(f"[{i}] decoded: {decoded!r}")

    # print("\nBatch examples:")
    # batch_ids = tokenizer.encode_batch(texts)
    # batch_decoded = tokenizer.decode_batch(batch_ids, skip_special_tokens=skip_special_tokens)
    # for i, (ids, decoded) in enumerate(zip(batch_ids, batch_decoded, strict=True), start=1):
    #     print(f"\n[{i}] ids:     {ids}")
    #     print(f"[{i}] decoded: {decoded!r}")

    ids = [1430]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(decoded)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
