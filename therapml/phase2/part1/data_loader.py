from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer.bpe import BPETokenizer


def _iter_texts_from_txt(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8")
    # TinyStories is typically one story per record; this keeps paragraphs together when blank-line separated.
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if chunks:
        yield from chunks
        return
    # Fallback: non-empty lines.
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def _iter_texts_from_jsonl(path: Path, *, text_field: str = "text") -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                raise KeyError(f"Missing field {text_field!r} in JSONL record")
            yield str(obj[text_field])


def _iter_texts_from_json(path: Path, *, text_field: str = "text") -> Iterable[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                if text_field not in item:
                    raise KeyError(f"Missing field {text_field!r} in JSON list record")
                yield str(item[text_field])
            else:
                raise TypeError(f"Unsupported JSON list item type: {type(item).__name__}")
        return
    if isinstance(obj, dict):
        if text_field in obj and isinstance(obj[text_field], list):
            for t in obj[text_field]:
                yield str(t)
            return
        raise TypeError(
            "Unsupported JSON object shape. Expected a list of texts/records, "
            f"or an object with a {text_field!r} list."
        )
    raise TypeError(f"Unsupported JSON root type: {type(obj).__name__}")


def iter_texts(path: str | Path, *, text_field: str = "text") -> Iterable[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in (".txt",):
        return _iter_texts_from_txt(path)
    if suffix in (".jsonl",):
        return _iter_texts_from_jsonl(path, text_field=text_field)
    if suffix in (".json",):
        return _iter_texts_from_json(path, text_field=text_field)

    raise ValueError(f"Unsupported dataset file type: {suffix} (expected .txt/.jsonl/.json)")


def build_token_stream(
    tokenizer: BPETokenizer,
    texts: Sequence[str],
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    max_tokens: int | None = None,
) -> list[int]:
    bos_id = tokenizer.token_to_id("<bos>") if add_bos else None
    eos_id = tokenizer.token_to_id("<eos>") if add_eos else None

    token_ids: list[int] = []
    for text in texts:
        ids: list[int] = []
        if bos_id is not None:
            ids.append(int(bos_id))
        ids.extend(tokenizer.encode(text))
        if eos_id is not None:
            ids.append(int(eos_id))

        token_ids.extend(ids)

        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break

    return token_ids


class TokenStreamDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Turns a 1D token stream into fixed-length next-token prediction examples.

    Each sample is:
      input_ids:  [seq_len]
      target_ids: [seq_len]  (input shifted by 1)
    """

    def __init__(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        seq_len: int,
        stride: int = 1,
        dtype: torch.dtype = torch.long,
    ) -> None:
        if int(seq_len) <= 0:
            raise ValueError("seq_len must be > 0")
        if int(stride) <= 0:
            raise ValueError("stride must be > 0")

        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids.to(dtype=dtype)
        else:
            tokens = torch.tensor(list(map(int, token_ids)), dtype=dtype)

        self.tokens = tokens
        self.seq_len = int(seq_len)
        self.stride = int(stride)

    def __len__(self) -> int:
        # Need seq_len + 1 tokens to create (x, y) pairs.
        n = int(self.tokens.numel())
        if n <= self.seq_len:
            return 0
        return 1 + (n - (self.seq_len + 1)) // self.stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(idx) * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        if chunk.numel() != self.seq_len + 1:
            raise IndexError("Index out of range for dataset windowing.")
        return chunk[:-1], chunk[1:]


@dataclass(frozen=True, slots=True)
class TinyStoriesLoaderConfig:
    seq_len: int = 128
    stride: int = 1
    batch_size: int = 8
    num_workers: int = 0

    train_split: float = 0.95
    seed: int = 1337

    max_train_texts: int | None = None
    max_eval_texts: int | None = None
    max_train_tokens: int | None = None
    max_eval_tokens: int | None = None

    add_bos: bool = True
    add_eos: bool = True


class TinyStoriesDataLoader:
    """
    Convenience wrapper to build train/eval torch DataLoaders for TinyStories-style text corpora.

    Dataset file formats supported: .txt, .jsonl, .json
    """

    def __init__(self, tokenizer: BPETokenizer, cfg: TinyStoriesLoaderConfig) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg

    def load_from_texts(self, texts: Sequence[str]) -> tuple[DataLoader, DataLoader]:
        if not texts:
            raise ValueError("No texts provided.")

        texts = list(texts)
        rng = random.Random(int(self.cfg.seed))
        rng.shuffle(texts)

        split_idx = max(1, int(len(texts) * float(self.cfg.train_split)))
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        if len(eval_texts) < 2:
            eval_texts = texts[: max(2, min(8, len(texts)))]

        if self.cfg.max_train_texts is not None:
            train_texts = train_texts[: int(self.cfg.max_train_texts)]
        if self.cfg.max_eval_texts is not None:
            eval_texts = eval_texts[: int(self.cfg.max_eval_texts)]

        train_tokens = build_token_stream(
            self.tokenizer,
            train_texts,
            add_bos=self.cfg.add_bos,
            add_eos=self.cfg.add_eos,
            max_tokens=self.cfg.max_train_tokens,
        )
        eval_tokens = build_token_stream(
            self.tokenizer,
            eval_texts,
            add_bos=self.cfg.add_bos,
            add_eos=self.cfg.add_eos,
            max_tokens=self.cfg.max_eval_tokens,
        )

        train_ds = TokenStreamDataset(train_tokens, seq_len=self.cfg.seq_len, stride=self.cfg.stride)
        eval_ds = TokenStreamDataset(eval_tokens, seq_len=self.cfg.seq_len, stride=self.cfg.stride)
        if len(eval_ds) == 0:
            # For very small corpora, the eval split may not have enough tokens to form even one window.
            eval_texts = texts[: max(2, min(8, len(texts)))]
            eval_tokens = build_token_stream(
                self.tokenizer,
                eval_texts,
                add_bos=self.cfg.add_bos,
                add_eos=self.cfg.add_eos,
                max_tokens=self.cfg.max_eval_tokens,
            )
            eval_ds = TokenStreamDataset(eval_tokens, seq_len=self.cfg.seq_len, stride=self.cfg.stride)

        batch_size = int(self.cfg.batch_size)
        drop_last_train = len(train_ds) >= batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            drop_last=drop_last_train,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(self.cfg.num_workers),
            drop_last=False,
        )
        return train_loader, eval_loader

    def load_from_train_eval_texts(
        self, train_texts: Sequence[str], eval_texts: Sequence[str]
    ) -> tuple[DataLoader, DataLoader]:
        if not train_texts:
            raise ValueError("No train_texts provided.")

        train_texts = list(train_texts)
        eval_texts = list(eval_texts)
        if len(eval_texts) < 2:
            eval_texts = train_texts[: max(2, min(8, len(train_texts)))]

        if self.cfg.max_train_texts is not None:
            train_texts = train_texts[: int(self.cfg.max_train_texts)]
        if self.cfg.max_eval_texts is not None:
            eval_texts = eval_texts[: int(self.cfg.max_eval_texts)]

        train_tokens = build_token_stream(
            self.tokenizer,
            train_texts,
            add_bos=self.cfg.add_bos,
            add_eos=self.cfg.add_eos,
            max_tokens=self.cfg.max_train_tokens,
        )
        eval_tokens = build_token_stream(
            self.tokenizer,
            eval_texts,
            add_bos=self.cfg.add_bos,
            add_eos=self.cfg.add_eos,
            max_tokens=self.cfg.max_eval_tokens,
        )

        train_ds = TokenStreamDataset(train_tokens, seq_len=self.cfg.seq_len, stride=self.cfg.stride)
        eval_ds = TokenStreamDataset(eval_tokens, seq_len=self.cfg.seq_len, stride=self.cfg.stride)

        batch_size = int(self.cfg.batch_size)
        drop_last_train = len(train_ds) >= batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            drop_last=drop_last_train,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(self.cfg.num_workers),
            drop_last=False,
        )
        return train_loader, eval_loader

    def load_from_file(self, path: str | Path, *, text_field: str = "text") -> tuple[DataLoader, DataLoader]:
        texts = list(iter_texts(path, text_field=text_field))
        return self.load_from_texts(texts)
