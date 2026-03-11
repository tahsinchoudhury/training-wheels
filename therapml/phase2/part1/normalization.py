import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, gamma=None, beta=None, eps=1e-8):
        super().__init__()
        if gamma is None:
            gamma = torch.ones(hidden_dim)
        if beta is None:
            beta = torch.zeros(hidden_dim)

        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = torch.square(x - mean).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance)

        normalized_x = (x - mean) / (std + self.eps)

        return self.gamma * normalized_x + self.beta
    
class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, gamma=None, eps=1e-8):
        super().__init__()
        if gamma is None:
            gamma = torch.ones(hidden_dim)

        self.gamma = nn.Parameter(gamma)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        variance = torch.square(x).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance)

        normalized_x = x / (std + self.eps)

        return self.gamma * normalized_x