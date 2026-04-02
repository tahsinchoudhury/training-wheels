from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@dataclass(frozen=True, slots=True)
class BPETrainingConfig:
    vocab_size: int = 8_000
    min_frequency: int = 2

    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    add_prefix_space: bool = True
    lowercase: bool = False

    def special_tokens(self) -> list[str]:
        # Keep ordering stable: trainer uses this list to reserve ids.
        return [self.unk_token, self.pad_token, self.bos_token, self.eos_token]


class BPETokenizer:
    """
    Lightweight wrapper around HuggingFace `tokenizers` providing:
    - training a ByteLevel BPE tokenizer from an iterator of texts
    - encode/decode (tokenize/detokenize)
    - save/load
    """

    def __init__(self, tokenizer: "Tokenizer"):
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> str | None:
        return self._tokenizer.id_to_token(int(idx))

    def encode(self, text: str) -> list[int]:
        enc = self._tokenizer.encode(text)
        return list(enc.ids)

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(list(texts))
        return [list(e.ids) for e in encodings]

    def decode_batch(self, batch_ids: Sequence[Sequence[int]], *, skip_special_tokens: bool = True) -> list[str]:
        return [
            self._tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens) for ids in batch_ids
        ]

    def save(self, path: str | Path) -> Path:
        """
        Saves a `tokenizers` JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        *,
        config: BPETrainingConfig | None = None,
    ) -> "BPETokenizer":
        """
        Trains a ByteLevel BPE tokenizer from an iterator of raw text strings.
        """
        if config is None:
            config = BPETrainingConfig()

        tokenizer = Tokenizer(BPE(unk_token=config.unk_token))

        if config.lowercase:
            from tokenizers.normalizers import Lowercase

            tokenizer.normalizer = NormalizerSequence([NFKC(), Lowercase()])
        else:
            tokenizer.normalizer = NormalizerSequence([NFKC()])

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=config.add_prefix_space)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=int(config.vocab_size),
            min_frequency=int(config.min_frequency),
            special_tokens=config.special_tokens(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)
