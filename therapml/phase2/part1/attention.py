import math
import torch
from torch import nn
from torch import Tensor

class Attention(nn.Module):

    @staticmethod
    def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)

        return torch.matmul(attn, V)
    
    def forward(self):
        pass