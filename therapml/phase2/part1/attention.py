import math
import torch
from torch import nn
from torch import Tensor
from . import activations

class Attention(nn.Module):

    @staticmethod
    def scaled_dot_product_attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None
    ):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = activations.Softmax()(x=scores, dim=-1)

        return torch.matmul(attn, V)
    
    def forward(self):
        pass