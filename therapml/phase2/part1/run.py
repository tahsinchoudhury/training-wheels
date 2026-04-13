from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset

from therapml.phase2.part1.data_loader import TinyStoriesDataLoader, TinyStoriesLoaderConfig
from therapml.phase2.part1.models.llama import Llama, LlamaConfig
from therapml.phase2.part1.train import TrainConfig, Trainer
from therapml.phase2.part1.tokenizer.bpe import BPETokenizer, BPETrainingConfig
from therapml.phase2.part1.logger import TrainingLogger


def _default_texts() -> list[str]:
    # Small built-in corpus so `run.py` works even without downloading TinyStories.
    return [
        "Once upon a time, a little cat found a shiny coin in the grass.",
        "Mia saw a red kite stuck in a tree. She asked her dad for help.",
        "Tom built a tiny robot from boxes. The robot waved hello and made Tom laugh.",
        "A brave turtle wanted to see the ocean. It walked slowly but never stopped.",
        "Lina planted a seed. Every day she gave it water. Soon, a green sprout appeared.",
        "Ben shared his toy car with Ava. Ava smiled and said thank you.",
        "A small dragon sneezed and made tiny sparks. It learned to breathe gently.",
        "Sam forgot his umbrella. A kind stranger shared theirs until the rain stopped.",
        "Nora baked cookies with her grandma. The kitchen smelled sweet and warm.",
        "A puppy found a lost mitten. The puppy carried it back to the door.",
    ]


def _load_or_train_tokenizer(tokenizer_path: Path, logger) -> BPETokenizer:
    if tokenizer_path.exists():
        return BPETokenizer.load(tokenizer_path)

    logger.info(f"Tokenizer not found at {tokenizer_path}; training a tiny tokenizer from built-in sample texts.")
    config = BPETrainingConfig(vocab_size=2000, min_frequency=1)
    tokenizer = BPETokenizer.train_from_iterator(_default_texts(), config=config)
    tokenizer.save(tokenizer_path)
    return tokenizer


def _load_tinystories_from_hf(
    *,
    text_field: str,
    max_train_texts: int,
    max_eval_texts: int,
    logger,
    seed=42,
) -> tuple[list[str], list[str]]:
    ds = load_dataset("roneneldan/TinyStories")
    if "train" not in ds or "validation" not in ds:
        raise KeyError(f"Expected 'train' and 'validation' splits, got: {list(ds.keys())}")

    train_ds = ds["train"].shuffle(seed=seed)
    eval_ds = ds["validation"].shuffle(seed=seed)
    if text_field not in train_ds.column_names:
        raise KeyError(f"Missing field {text_field!r} in train split. Available columns: {train_ds.column_names}")
    if text_field not in eval_ds.column_names:
        raise KeyError(
            f"Missing field {text_field!r} in validation split. Available columns: {eval_ds.column_names}"
        )
    
    logger.info(f"Loaded dataset with {len(train_ds)} train texts and {len(eval_ds)} validation texts.")

    train_take = min(max_train_texts, len(train_ds))
    eval_take = min(max_eval_texts, len(eval_ds))
    
    # train_texts = [str(t) for t in train_ds.select(range(train_take))[text_field] if str(t).strip()]
    # eval_texts = [str(t) for t in eval_ds.select(range(eval_take))[text_field] if str(t).strip()]

    train_texts = [
        str(x[text_field]).strip()
        for x in train_ds.take(train_take)
        if x[text_field] and str(x[text_field]).strip()
    ]

    eval_texts = [
        str(x[text_field]).strip()
        for x in eval_ds.take(eval_take)
        if x[text_field] and str(x[text_field]).strip()
    ]

    logger.info(f"Using {len(train_texts)} train texts and {len(eval_texts)} validation texts after filtering empty/whitespace-only entries.")    
    
    return train_texts, eval_texts


def main() -> int:
    logger = TrainingLogger.get_logger(__name__)
    parser = argparse.ArgumentParser(description="TinyStories LM runner (train/eval + quick inference).")
    parser.add_argument("--tokenizer", default="data/tinystories_bpe.json", help="Path to tokenizer JSON.")
    parser.add_argument(
        "--dataset",
        default="",
        help=(
            "Optional dataset file (.txt/.jsonl/.json). If omitted, loads HuggingFace dataset "
            "roneneldan/TinyStories (falls back to a tiny built-in corpus if download fails)."
        ),
    )
    parser.add_argument("--text-field", default="text", help="JSON/JSONL text field name.")

    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--generation-step-interval", type=int, default=500)
    parser.add_argument("--plot-interval-steps", type=int, default=5000, help="Plot losses every N steps (0 to disable).")
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=0, help="Save checkpoint every N epochs (0 to disable).")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints", help="Directory to save model checkpoints.")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--max-train-texts", type=int, default=50)
    parser.add_argument("--max-eval-texts", type=int, default=20)
    parser.add_argument("--max-train-tokens", type=int, default=6000)
    parser.add_argument("--max-eval-tokens", type=int, default=2000)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-model", default=None, help="Path to a pre-trained model checkpoint to load (optional).")
    parser.add_argument("--save-model", default="models/llama_phase2_part1.pth", help="Path where to save the trained model.")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    tokenizer = _load_or_train_tokenizer(tokenizer_path, logger)
    logger.info(f"Loaded tokenizer vocab_size={tokenizer.vocab_size} from {tokenizer_path}")

    loader_cfg = TinyStoriesLoaderConfig(
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        stride=int(args.stride),
        train_split=0.95,
        max_train_texts=int(args.max_train_texts),
        max_eval_texts=int(args.max_eval_texts),
        max_train_tokens=int(args.max_train_tokens),
        max_eval_tokens=int(args.max_eval_tokens),
    )
    data_loader = TinyStoriesDataLoader(tokenizer, loader_cfg)

    if args.dataset:
        train_loader, eval_loader = data_loader.load_from_file(args.dataset, text_field=args.text_field)
        logger.info(f"Loaded dataset from {args.dataset}")
    else:
        try:
            train_texts, eval_texts = _load_tinystories_from_hf(
                text_field=args.text_field,
                max_train_texts=int(args.max_train_texts),
                max_eval_texts=int(args.max_eval_texts),
                logger=logger,
            )
            train_loader, eval_loader = data_loader.load_from_train_eval_texts(train_texts, eval_texts)
            logger.info(
                "Loaded HuggingFace dataset roneneldan/TinyStories "
                f"(train={len(train_texts)} texts, validation={len(eval_texts)} texts)."
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset roneneldan/TinyStories ({type(e).__name__}: {e}).")
            logger.info("Falling back to built-in tiny corpus.")
            train_loader, eval_loader = data_loader.load_from_texts(_default_texts())

    if len(train_loader) == 0 or len(eval_loader) == 0:
        raise ValueError(
            "Train/eval loaders are empty. Try decreasing --seq-len / --batch-size "
            "or increasing --max-*-tokens."
        )

    logger.debug(f"Before training, memory allocated: {torch.cuda.memory_allocated(args.device) / 1e6:.2f}MB, reserved: {torch.cuda.memory_reserved(args.device) / 1e6:.2f}MB")

    # Load or create model
    if args.load_model:
        logger.info(f"Loading pre-trained model from {args.load_model}")
        model = Llama.load(args.load_model)
    else:
        model_cfg = LlamaConfig(
            vocab_size=int(tokenizer.vocab_size),
            context_length=int(args.seq_len),
            d_model=int(args.d_model),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            d_ff=int(args.d_ff),
            rope_theta=float(args.rope_theta),
        )
        model = Llama(model_cfg)
    
    total_params, trainable_params = model.num_parameters()
    logger.info(f"Initialized Llama model with {total_params / 1e6:.2f}M total parameters, {trainable_params / 1e6:.2f}M trainable parameters.")

    device = torch.device(args.device)
    model.to(device)

    logger.debug(f"After initializing the model, memory allocated: {torch.cuda.memory_allocated(args.device) / 1e6:.2f}MB, reserved: {torch.cuda.memory_reserved(args.device) / 1e6:.2f}MB")

    train_cfg = TrainConfig(
        num_epochs=int(args.epochs),
        max_lr=float(args.max_lr),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        generation_step_interval=int(args.generation_step_interval),
        plot_interval_steps=int(args.plot_interval_steps),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        checkpoint_dir=str(args.checkpoint_dir),
        device=str(args.device),
    )

    trainer = Trainer(
        log_every_n_steps=10,
        eval_log_every_n_batches=50,
    )
    history = trainer.train(
        model=model,
        train_cfg=train_cfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
    )
    logger.info(f"Training complete. History: {history}")

    # Save the trained model
    logger.info(f"Saving trained model to {args.save_model}")
    model.save(args.save_model)
    logger.info(f"Model saved successfully to {args.save_model}")

    # Load the model in a new object to test save/load functionality
    logger.info(f"Testing save/load by loading model from {args.save_model}")
    trained_model = Llama.load(args.save_model)
    trained_model.to(device)
    logger.info("Model successfully loaded and moved to device")

    logger.info("Inference samples:")
    prompts = [
        "Once upon a time",
        "Mia saw",
        "A small dragon",
        "It has been an absolute",
    ]
    eos_id = tokenizer.token_to_id("<eos>")
    for p in prompts:
        input_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        generated_ids = trained_model.generate(
            input_ids,
            device=device,
            max_new_tokens=40,
            eos_id=eos_id,
        )
        out = tokenizer.decode(generated_ids[0].tolist())
        logger.info(f"Prompt: {p!r}\nOutput:  {out!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
