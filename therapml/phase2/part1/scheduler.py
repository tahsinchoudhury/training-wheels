import math
import torch

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

        # Start from 0 or min_lr. Usually 0 is fine for warmup.
        self._set_lr(0.0 if warmup_steps > 0 else max_lr)

    def _set_lr(
        self,
        lr: float
    ) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(
        self,
        step: int,
    ) -> float:
        if step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        
        if step > self.total_steps:
            return self.min_lr
        
        decay_steps = self.total_steps - self.warmup_steps
        decay_steps_taken = step - self.warmup_steps
        cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_steps_taken))

        return self.min_lr + (self.max_lr - self.min_lr) * cosine_coeff
    
    def step(
        self,
    ) -> float:
        self.step_num += 1
        lr = self._get_lr(self.step_num)
        self._set_lr(lr)
        return lr