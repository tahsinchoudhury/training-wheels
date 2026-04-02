from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    num_epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    eval_every_epoch: bool = True
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device(config.device)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.global_step = 0

        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit(self) -> None:
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            train_metrics = self.train_one_epoch()
            self._log_epoch_metrics("train", train_metrics)

            val_metrics = None
            if self.val_loader is not None and self.config.eval_every_epoch:
                val_metrics = self.validate()
                self._log_epoch_metrics("val", val_metrics)

            self._step_scheduler(val_metrics)
            self._save_checkpoint(val_metrics)

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()

        running_loss = 0.0
        total_samples = 0
        correct = 0

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            inputs, targets = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            correct += self._count_correct(outputs, targets)

            self.global_step += 1

            if batch_idx % self.config.log_every == 0:
                avg_loss = running_loss / total_samples
                avg_acc = correct / total_samples
                print(
                    f"[Epoch {self.current_epoch}/{self.config.num_epochs}] "
                    f"[Step {batch_idx}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
                )

        return {
            "loss": running_loss / total_samples,
            "accuracy": correct / total_samples,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()

        running_loss = 0.0
        total_samples = 0
        correct = 0

        for batch in self.val_loader:
            inputs, targets = self._move_batch_to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            correct += self._count_correct(outputs, targets)

        return {
            "loss": running_loss / total_samples,
            "accuracy": correct / total_samples,
        }

    def _move_batch_to_device(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assumes batch is either:
        1. (inputs, targets)
        2. {"input": ..., "target": ...}
        3. {"input_ids": ..., "labels": ...}
        Adjust this method to match your dataset output format.
        """
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        elif isinstance(batch, dict):
            if "input" in batch and "target" in batch:
                inputs, targets = batch["input"], batch["target"]
            elif "input_ids" in batch and "labels" in batch:
                inputs, targets = batch["input_ids"], batch["labels"]
            else:
                raise KeyError(
                    "Unsupported batch dictionary format. "
                    "Expected keys like ('input', 'target') or ('input_ids', 'labels')."
                )
        else:
            raise TypeError("Unsupported batch type.")

        return inputs.to(self.device), targets.to(self.device)

    def _count_correct(self, outputs: torch.Tensor, targets: torch.Tensor) -> int:
        """
        For classification.
        For regression or language modeling, replace this logic.
        """
        preds = outputs.argmax(dim=1)
        return (preds == targets).sum().item()

    def _step_scheduler(self, val_metrics: Optional[dict[str, float]]) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_metrics is None:
                return
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()

    def _save_checkpoint(self, val_metrics: Optional[dict[str, float]]) -> None:
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        latest_path = self.save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        if val_metrics is None:
            return

        current_val_loss = val_metrics["loss"]
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            checkpoint["best_val_loss"] = self.best_val_loss
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved new best checkpoint to {best_path}")

        elif not self.config.save_best_only:
            epoch_path = self.save_dir / f"epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, epoch_path)

    def _log_epoch_metrics(self, split: str, metrics: dict[str, float]) -> None:
        metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[Epoch {self.current_epoch}] {split}: {metric_str}")