# from . import dropout, loss
import torch

# scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
mask = ~mask
scores = torch.Tensor([2, 1, 3, 4])
scores = scores.masked_fill(mask, -5)
print(scores)

# class Attention(nn.Module):

#     @staticmethod
#     def scaled_dot_product_attention(
#         Q: Tensor,
#         K: Tensor,
#         V: Tensor,
#         mask: Tensor | None = None,
#     ) -> Tensor:

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

#         if mask is not None:
#             scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

#         attn = torch.softmax(scores, dim=-1)

#         return torch.matmul(attn, V)
    
#     def forward(self):
#         pass