from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from therapml.phase2.part1.logger import TrainingLogger
from therapml.phase2.part1.models.llama import Llama
from therapml.phase2.part1.optimizers import AdamW
from therapml.phase2.part1.scheduler import WarmupCosineScheduler
from therapml.phase2.part1.tokenizer.bpe import BPETokenizer


ANSWER_LETTERS = ("A", "B", "C", "D")
DEFAULT_CHECKPOINT_PATH = "models/checkpoints/checkpoint_epoch_5.pth"
DEFAULT_TOKENIZER_PATH = "data/tinystories_bpe.json"
DEFAULT_OUTPUT_PATH = "models/checkpoints/mmlu_finetuned.pth"
DEFAULT_CONTEXT_LENGTH = 1024
IGNORE_INDEX = -100


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def format_mmlu_prompt(example: dict[str, Any]) -> str:
    question = str(example["question"]).strip()
    choices = list(example["choices"])
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")

    return (
        f"{question}\n"
        f"A. {str(choices[0]).strip()}\n"
        f"B. {str(choices[1]).strip()}\n"
        f"C. {str(choices[2]).strip()}\n"
        f"D. {str(choices[3]).strip()}\n"
        "Answer:"
    )


def format_answer(example: dict[str, Any]) -> str:
    answer_idx = int(example["answer"])
    if answer_idx < 0 or answer_idx >= len(ANSWER_LETTERS):
        raise ValueError(f"Answer index must be in [0, 3], got {answer_idx}")
    return f" {ANSWER_LETTERS[answer_idx]}"


@dataclass(frozen=True, slots=True)
class EncodedExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class MMLUAnswerDataset(Dataset[EncodedExample]):
    """MMLU examples encoded for answer-only causal LM fine-tuning."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]] | HFDataset,
        tokenizer: BPETokenizer,
        *,
        seq_len: int,
        include_eos: bool = True,
        answer_only_loss: bool = True,
        show_progress: bool = True,
    ) -> None:
        if int(seq_len) <= 0:
            raise ValueError("seq_len must be > 0")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.include_eos = bool(include_eos)
        self.answer_only_loss = bool(answer_only_loss)
        self.eos_id = tokenizer.token_to_id("<eos>")
        if self.include_eos and self.eos_id is None:
            raise ValueError("Tokenizer is missing required special token '<eos>'")

        iterable = rows
        total = len(rows)
        if show_progress:
            iterable = tqdm(rows, total=total, desc="Tokenizing MMLU")

        examples: list[EncodedExample] = []
        skipped = 0
        for row in iterable:
            try:
                examples.append(self._encode_one(dict(row)))
            except ValueError:
                skipped += 1

        if not examples:
            raise ValueError("No usable MMLU examples were encoded.")

        self.examples = examples
        self.skipped = skipped

    def _encode_one(self, row: dict[str, Any]) -> EncodedExample:
        prompt_ids = self.tokenizer.encode(format_mmlu_prompt(row))
        answer_ids = self.tokenizer.encode(format_answer(row))
        if not answer_ids:
            raise ValueError("Answer encoded to an empty token sequence")

        full_ids = prompt_ids + answer_ids
        if self.include_eos:
            full_ids.append(int(self.eos_id))

        if len(full_ids) < 2:
            raise ValueError("Encoded sequence is too short for next-token prediction")

        # Keep the supervised answer at the end, matching the eval wrapper's left-truncation behavior.
        max_tokens = self.seq_len + 1
        overflow = max(0, len(full_ids) - max_tokens)
        if overflow:
            full_ids = full_ids[overflow:]

        prompt_len_after_truncation = max(0, len(prompt_ids) - overflow)
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        if self.answer_only_loss:
            first_supervised_label = max(0, prompt_len_after_truncation - 1)
            if first_supervised_label > 0:
                labels[:first_supervised_label] = IGNORE_INDEX

        if bool(torch.all(labels == IGNORE_INDEX)):
            raise ValueError("Example has no supervised target tokens after truncation")

        return EncodedExample(input_ids=input_ids, labels=labels)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[int(idx)]


@dataclass(frozen=True, slots=True)
class MMLUBatchCollator:
    pad_token_id: int

    def __call__(self, examples: list[EncodedExample]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(example.input_ids.numel() for example in examples)
        input_ids = torch.full(
            (len(examples), max_len),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(examples), max_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )

        for idx, example in enumerate(examples):
            length = example.input_ids.numel()
            input_ids[idx, :length] = example.input_ids
            labels[idx, :length] = example.labels

        return input_ids, labels


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def _maybe_select_split(
    split: HFDataset,
    *,
    max_examples: int | None,
    seed: int,
    shuffle: bool,
) -> HFDataset:
    if shuffle:
        split = split.shuffle(seed=int(seed))
    if max_examples is not None:
        n = min(int(max_examples), len(split))
        split = split.select(range(n))
    return split


def _build_loader(
    split: HFDataset,
    tokenizer: BPETokenizer,
    *,
    seq_len: int,
    batch_size: int,
    include_eos: bool,
    answer_only_loss: bool,
    num_workers: int,
    pin_memory: bool,
    shuffle_batches: bool,
) -> DataLoader:
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing required special token '<pad>'")

    dataset = MMLUAnswerDataset(
        split,
        tokenizer,
        seq_len=int(seq_len),
        include_eos=include_eos,
        answer_only_loss=answer_only_loss,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_batches),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=MMLUBatchCollator(pad_token_id=int(pad_token_id)),
        drop_last=False,
    )


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        input_ids = input_ids.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        loss = _compute_loss(model(input_ids), labels)
        total_loss += float(loss.item())
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    return avg_loss, perplexity


def fine_tune(args: argparse.Namespace) -> Path:
    logger = TrainingLogger.get_logger(__name__)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = _resolve_device(args.device)
    logger.info(f"Using device={device}")

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    model = Llama.load(args.checkpoint_path, map_location=device)
    if int(tokenizer.vocab_size) != int(model.vocab_size):
        raise ValueError(
            f"Tokenizer/model vocab mismatch: tokenizer={tokenizer.vocab_size}, model={model.vocab_size}"
        )

    context_length = int(args.context_length)
    if context_length != int(model.context_length):
        logger.info(f"Updating model context_length from {model.context_length} to {context_length}")
        model.set_context_length(context_length)

    seq_len = int(args.seq_len) if args.seq_len is not None else int(model.context_length)
    if seq_len > int(model.context_length):
        raise ValueError(f"--seq-len ({seq_len}) cannot exceed model context_length ({model.context_length})")

    logger.info(
        f"Loaded checkpoint={args.checkpoint_path} tokenizer={args.tokenizer_path} "
        f"context_length={model.context_length} seq_len={seq_len}"
    )

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if args.train_split not in dataset:
        raise KeyError(f"Missing train split {args.train_split!r}; available splits: {list(dataset.keys())}")
    if args.eval_split and args.eval_split not in dataset:
        raise KeyError(f"Missing eval split {args.eval_split!r}; available splits: {list(dataset.keys())}")

    train_split = _maybe_select_split(
        dataset[args.train_split],
        max_examples=args.max_train_examples,
        seed=int(args.seed),
        shuffle=not args.no_shuffle,
    )
    eval_split = None
    if args.eval_split:
        eval_split = _maybe_select_split(
            dataset[args.eval_split],
            max_examples=args.max_eval_examples,
            seed=int(args.seed) + 1,
            shuffle=False,
        )

    logger.info(
        f"Using dataset={args.dataset_name}/{args.dataset_config} "
        f"train_split={args.train_split} train_rows={len(train_split)} "
        f"eval_split={args.eval_split or 'none'} eval_rows={len(eval_split) if eval_split is not None else 0}"
    )

    train_loader = _build_loader(
        train_split,
        tokenizer,
        seq_len=seq_len,
        batch_size=int(args.batch_size),
        include_eos=not args.no_eos,
        answer_only_loss=not args.full_sequence_loss,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        shuffle_batches=True,
    )
    eval_loader = None
    if eval_split is not None:
        eval_loader = _build_loader(
            eval_split,
            tokenizer,
            seq_len=seq_len,
            batch_size=int(args.eval_batch_size or args.batch_size),
            include_eos=not args.no_eos,
            answer_only_loss=not args.full_sequence_loss,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            shuffle_batches=False,
        )

    logger.info(f"Built train_loader={len(train_loader)} batches")
    if eval_loader is not None:
        logger.info(f"Built eval_loader={len(eval_loader)} batches")

    model.to(device)
    model.train()

    total_steps = max(1, int(args.epochs) * len(train_loader))
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.max_lr),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.eps),
        weight_decay=float(args.weight_decay),
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=int(args.warmup_steps),
        total_steps=total_steps,
        max_lr=float(args.max_lr),
        min_lr=float(args.min_lr),
    )

    global_step = 0
    for epoch in range(int(args.epochs)):
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Fine-tuning epoch {epoch + 1}/{args.epochs}")
        model.train()

        for batch_idx, (input_ids, labels) in enumerate(progress):
            global_step += 1
            input_ids = input_ids.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = _compute_loss(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            lr = scheduler.step()

            loss_value = float(loss.item())
            running_loss += loss_value
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")

            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                avg = running_loss / max(1, batch_idx + 1)
                logger.info(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={loss_value:.4f} avg_epoch_loss={avg:.4f} "
                    f"grad_norm={float(grad_norm):.4f} lr={lr:.8f}"
                )

            if (
                eval_loader is not None
                and int(args.eval_every_steps) > 0
                and global_step % int(args.eval_every_steps) == 0
            ):
                eval_loss, eval_ppl = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    max_batches=args.max_eval_batches,
                )
                logger.info(f"step={global_step} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")
                model.train()

        avg_epoch_loss = running_loss / max(1, len(train_loader))
        logger.info(f"epoch={epoch + 1}/{args.epochs} train_loss={avg_epoch_loss:.4f}")

        if eval_loader is not None:
            eval_loss, eval_ppl = evaluate(
                model,
                eval_loader,
                device=device,
                max_batches=args.max_eval_batches,
            )
            logger.info(f"epoch={epoch + 1}/{args.epochs} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")

        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = checkpoint_dir / f"mmlu_finetuned_epoch_{epoch + 1}.pth"
            model.save(epoch_path)
            logger.info(f"Saved epoch checkpoint to {epoch_path}")

    output_path = Path(args.output_path)
    model.save(output_path)
    logger.info(f"Saved fine-tuned model to {output_path}")
    return output_path


def _optional_int(value: str) -> int | None:
    if value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the local TherapML Llama checkpoint on MMLU auxiliary_train.")

    parser.add_argument("--dataset-name", default="cais/mmlu")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument("--train-split", default="auxiliary_train")
    parser.add_argument("--eval-split", default="validation", help="Set to '' to disable validation loss.")
    parser.add_argument("--max-train-examples", type=_optional_int, default=None)
    parser.add_argument("--max-eval-examples", type=_optional_int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-shuffle", action="store_true")

    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", default="", help="Optional directory for per-epoch fine-tune checkpoints.")

    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to --context-length.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")

    parser.add_argument("--max-lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=_optional_int, default=None)
    parser.add_argument("--no-eos", action="store_true", help="Do not append/train the answer EOS token.")
    parser.add_argument(
        "--full-sequence-loss",
        action="store_true",
        help="Train on all prompt and answer tokens instead of masking prompt tokens.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.eval_split == "":
        args.eval_split = None
    fine_tune(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
