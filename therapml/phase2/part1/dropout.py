import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("Dropout probability p must satisfy 0 <= p <= 1.")
        self.p = p

    def forward(self, x: torch.Tensor):
        if self.training == False or self.p == 0:
            return x
        if self.p == 1:
            return torch.zeros_like(x)
        
        keep_prob = 1 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)

        return x * mask / keep_prob