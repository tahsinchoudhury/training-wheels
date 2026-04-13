from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from therapml.phase2.part1.tokenizer.bpe import BPETokenizer

from .scheduler import WarmupCosineScheduler
from .optimizers import AdamW
from .loss import CrossEntropyLoss
from .plotter import LossPlotter
from .logger import TrainingLogger


@dataclass
class TrainConfig:
    num_epochs: int = 5
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    generation_step_interval: int = 500
    plot_interval_steps: int = 5000  # 0 means plot only at end, >0 plots every N steps
    checkpoint_interval_epochs: int = 0  # 0 means no checkpoints, >0 saves every N epochs
    checkpoint_dir: str = "models/checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(
        self,
        log_every_n_steps: int = 100,
        eval_log_every_n_batches: int = 50,
    ):
        self.log_every_n_steps = int(log_every_n_steps)
        self.eval_log_every_n_batches = int(eval_log_every_n_batches)

    @staticmethod
    def _compute_loss(
        loss_fn: CrossEntropyLoss,
        logits: torch.Tensor, # (batch size, seq len, vocab size)
        targets: torch.Tensor, # (batch size, seq len) -- contains target ids
    ) -> torch.Tensor:
        if logits.ndim not in (2, 3):
            raise ValueError(f"Expected logits to be 2D or 3D, got shape {tuple(logits.shape)}")

        num_classes = logits.shape[-1]
        logits_2d = logits.reshape(-1, num_classes)
        targets_1d = targets.reshape(-1)
        return loss_fn(logits_2d, targets_1d)

    def evaluate(
        self,
        model: torch.nn.Module,
        eval_loader: DataLoader,
        device: torch.device,
        loss_fn: CrossEntropyLoss,
        *,
        epoch: int,
        num_epochs: int,
    ) -> tuple[float, float]:
        logger = TrainingLogger.get_logger(__name__)
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch_idx, batch in enumerate(eval_loader):
                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                loss = self._compute_loss(loss_fn, logits, target_ids)

                total_loss += float(loss.item())
                num_batches += 1

                if self.eval_log_every_n_batches > 0 and batch_idx % self.eval_log_every_n_batches == 0:
                    logger.info(
                        f"eval epoch={epoch}/{num_epochs} "
                        f"batch={batch_idx+1}/{len(eval_loader)} "
                        f"loss={loss.item():.4f}"
                    )

        avg_loss = total_loss / max(1, num_batches)
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")

        return avg_loss, perplexity

    def train(
        self,
        model: torch.nn.Module,
        train_cfg: TrainConfig,
        train_loader: DataLoader, 
        eval_loader: DataLoader,
        tokenizer: BPETokenizer | None = None,
    ):
        logger = TrainingLogger.get_logger(__name__)
        device = torch.device(train_cfg.device)
        model.to(device)

        total_steps = train_cfg.num_epochs * len(train_loader)

        optimizer = AdamW(
            params=model.parameters(),
            lr=train_cfg.max_lr,
            betas=train_cfg.betas,
            eps=train_cfg.eps,
            weight_decay=train_cfg.weight_decay,
        )

        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=total_steps,
            max_lr=train_cfg.max_lr,
            min_lr=train_cfg.min_lr,
        )

        all_train_losses = []  # global list of all training step losses
        train_losses = []      # averaged losses at plot intervals
        eval_losses = []       # averaged losses per epoch

        loss_fn = CrossEntropyLoss(target_type="indices")

        global_step = 0

        for epoch in range(train_cfg.num_epochs):
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1
                
                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                optimizer.zero_grad()

                logits = model(input_ids)
                
                loss = self._compute_loss(loss_fn, logits, target_ids)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=train_cfg.grad_clip,
                )

                optimizer.step()
                lr = scheduler.step()

                all_train_losses.append(float(loss.item()))

                # Generate examples at configured interval
                if train_cfg.generation_step_interval > 0 and global_step % train_cfg.generation_step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        # Use the first token from the batch as starting point
                        start_tokens = input_ids[:1, :1]  # [1, 1]
                        generated = model.generate(
                            start_tokens,
                            device=device,
                            max_new_tokens=100,
                        )
                    if tokenizer is not None:
                        logger.info(f"[Step {global_step}] Generated (length={generated.shape[1]}): {tokenizer.decode(generated[0].tolist())}...")
                    model.train()

                if self.log_every_n_steps > 0 and batch_idx % self.log_every_n_steps == 0:
                    logger.info(
                        f"epoch={epoch+1}/{train_cfg.num_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={loss.item():.4f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"lr={lr:.8f}"
                    )

                # Store average losses at configured interval
                if train_cfg.plot_interval_steps > 0 and global_step % train_cfg.plot_interval_steps == 0:
                    # Average last plot_interval_steps losses
                    start_idx = max(0, len(all_train_losses) - train_cfg.plot_interval_steps)
                    avg_interval_loss = sum(all_train_losses[start_idx:]) / max(1, len(all_train_losses) - start_idx)
                    train_losses.append(avg_interval_loss)
                    logger.info(f"[Step {global_step}] Interval train loss: {avg_interval_loss:.4f}")

                    # Evaluate at interval
                    if eval_loader is not None:
                        avg_eval_loss, eval_ppl = self.evaluate(
                            model=model,
                            eval_loader=eval_loader,
                            device=device,
                            loss_fn=loss_fn,
                            epoch=epoch + 1,
                            num_epochs=train_cfg.num_epochs,
                        )
                        eval_losses.append(avg_eval_loss)
                        logger.info(
                            f"[Step {global_step}] Interval eval loss: {avg_eval_loss:.4f} "
                            f"eval_ppl={eval_ppl:.2f}"
                        )

            # Save checkpoint if interval is configured
            if train_cfg.checkpoint_interval_epochs > 0 and (epoch + 1) % train_cfg.checkpoint_interval_epochs == 0:
                checkpoint_dir = Path(train_cfg.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                model.save(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Plot losses after training ends
        if train_losses or eval_losses:
            LossPlotter().plot_losses(
                train_losses=train_losses,
                eval_losses=eval_losses,
                out_path=Path("plots") / "phase2_part1_losses.png",
            )

        return {
            "train_losses": train_losses,
            "eval_losses": eval_losses,
        }
                
