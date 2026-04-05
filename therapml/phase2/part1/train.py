from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .scheduler import WarmupCosineScheduler
from .optimizers import AdamW
from .loss import CrossEntropyLoss
from .plotter import LossPlotter


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
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supports:
        - logits [B, T, V] with targets [B, T] (class indices) via torch CE
        - logits [B, T, V] with targets [B, T, V] (one-hot / soft) via custom CE
        - logits [B, V] with targets [B] (class indices) via torch CE
        - logits [B, V] with targets [B, V] (one-hot / soft) via custom CE
        """
        if logits.ndim not in (2, 3):
            raise ValueError(f"Expected logits to be 2D or 3D, got shape {tuple(logits.shape)}")

        num_classes = logits.shape[-1]
        logits_2d = logits.reshape(-1, num_classes)

        # Soft / one-hot labels
        if targets.shape == logits.shape:
            targets_2d = targets.reshape(-1, num_classes).to(dtype=logits_2d.dtype)
            return loss_fn(logits_2d, targets_2d)

        # Class-index labels
        if targets.ndim == logits.ndim - 1:
            targets_1d = targets.reshape(-1).to(dtype=torch.long)
            return F.cross_entropy(logits_2d, targets_1d)

        raise ValueError(
            f"Unsupported targets shape {tuple(targets.shape)} for logits shape {tuple(logits.shape)}"
        )

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
                    print(
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
    ):
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

        train_losses = []
        eval_losses = []

        loss_fn = CrossEntropyLoss()

        global_step = 0

        for epoch in range(train_cfg.num_epochs):
            model.train()

            train_loss_sum = 0.0

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

                train_loss_sum += float(loss.item())

                if self.log_every_n_steps > 0 and batch_idx % self.log_every_n_steps == 0:
                    print(
                        f"epoch={epoch+1}/{train_cfg.num_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={loss.item():.4f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"lr={lr:.8f}"
                    )
            
            avg_train_loss = train_loss_sum / max(1, len(train_loader))
            train_losses.append(avg_train_loss)

            if eval_loader is not None:
                avg_eval_loss, eval_ppl = self.evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    device=device,
                    loss_fn=loss_fn,
                    epoch=epoch + 1,
                    num_epochs=train_cfg.num_epochs,
                )
            else:
                avg_eval_loss, eval_ppl = float("nan"), float("nan")

            eval_losses.append(avg_eval_loss)

            print(
                f"epoch_end epoch={epoch+1}/{train_cfg.num_epochs} "
                f"train_loss={avg_train_loss:.4f} "
                f"eval_loss={avg_eval_loss:.4f} "
                f"eval_ppl={eval_ppl:.2f}"
            )

        LossPlotter().plot_losses(
            train_losses=train_losses,
            eval_losses=eval_losses,
            out_path=Path("plots") / "phase2_part1_losses.png",
        )

        return {
            "train_losses": train_losses,
            "eval_losses": eval_losses,
        }
                
