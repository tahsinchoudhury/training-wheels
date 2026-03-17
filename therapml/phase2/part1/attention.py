import math
import torch
from torch import nn
from torch import Tensor
from . import activations
from jaxtyping import Float, Int
from . import pos_embedding

class Attention(nn.Module):

    @staticmethod
    def scaled_dot_product_attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = activations.Softmax()(x=scores, dim=-1)

        return torch.matmul(attn, V)
    
    def forward(self):
        pass

class MultiHeadAttention(nn.Module):

    @staticmethod
    def multihead_attention(
        d_model,
        num_heads,
        q_proj_weight: Float[Tensor, "d_k d_in"],
        k_proj_weight: Float[Tensor, "d_k d_in"],
        v_proj_weight: Float[Tensor, "d_v d_in"],
        o_proj_weight: Float[Tensor, "d_model d_v"],
        in_features: Float[Tensor, "batch ctx_len d_in"],
        mask: torch.Tensor | None = None
    ):
        batch, ctx_len, d_in = in_features.shape
        d_k, d_in = q_proj_weight.shape
        d_v, d_in = v_proj_weight.shape
        dk_head = d_k // num_heads
        dv_head = d_v // num_heads

        Q = torch.matmul(in_features, q_proj_weight.T)
        K = torch.matmul(in_features, k_proj_weight.T)
        V = torch.matmul(in_features, v_proj_weight.T)

        Q = Q.view(batch, ctx_len, num_heads, dk_head).transpose(1, 2)
        K = K.view(batch, ctx_len, num_heads, dk_head).transpose(1, 2)
        V = V.view(batch, ctx_len, num_heads, dv_head).transpose(1, 2)

        # shape of mask: [batch, ctx_len, ctx_len] -> [batch, 1, ctx_len, ctx_len]
        if mask is not None:
            if mask.shape[-2:] != (ctx_len, ctx_len):
                mask = mask[..., :ctx_len, :ctx_len]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = Attention.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, ctx_len, d_v)

        return torch.matmul(out, o_proj_weight.T)
    
    @staticmethod
    def multihead_attention_with_rope(
        d_model: int,
        num_heads: int,
        ctx_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, "d_k d_in"],
        k_proj_weight: Float[Tensor, "d_k d_in"],
        v_proj_weight: Float[Tensor, "d_v d_in"],
        o_proj_weight: Float[Tensor, "d_model d_v"],
        in_features: Float[Tensor, "batch ctx_len d_in"],
        token_positions: Int[Tensor, "batch ctx_len"],
        mask: torch.Tensor | None = None,
    ):
        batch, seq_len, d_in = in_features.shape
        d_k, d_in = q_proj_weight.shape
        d_v, d_in = v_proj_weight.shape
        dk_head = d_k // num_heads
        dv_head = d_v // num_heads

        Q = torch.matmul(in_features, q_proj_weight.T)
        K = torch.matmul(in_features, k_proj_weight.T)
        V = torch.matmul(in_features, v_proj_weight.T)

        Q = Q.view(batch, seq_len, num_heads, dk_head).transpose(1, 2)
        K = K.view(batch, seq_len, num_heads, dk_head).transpose(1, 2)
        V = V.view(batch, seq_len, num_heads, dv_head).transpose(1, 2)

        # apply rope
        rope = pos_embedding.RoPE(embedding_dim=dk_head, theta=theta, context_len=ctx_len)

        # token_positions is typically shared across batch; expand as needed
        if token_positions.ndim == 1:
            token_positions = token_positions.unsqueeze(0)
        if token_positions.shape[0] == 1 and batch > 1:
            token_positions = token_positions.expand(batch, -1)
        if token_positions.shape != (batch, seq_len):
            raise ValueError(
                f"token_positions must have shape {(batch, seq_len)}, got {tuple(token_positions.shape)}"
            )

        token_positions_heads = (
            token_positions.unsqueeze(1)
            .expand(batch, num_heads, seq_len)
            .reshape(batch * num_heads, seq_len)
        )

        q_flat = Q.reshape(batch * num_heads, seq_len, dk_head)
        k_flat = K.reshape(batch * num_heads, seq_len, dk_head)
        q_flat = rope(q_flat, token_positions_heads)
        k_flat = rope(k_flat, token_positions_heads)
        Q = q_flat.reshape(batch, num_heads, seq_len, dk_head)
        K = k_flat.reshape(batch, num_heads, seq_len, dk_head)

        # shape of mask: [batch, ctx_len, ctx_len] -> [batch, 1, seq_len, seq_len]
        if mask is not None:
            if mask.shape[-2:] != (seq_len, seq_len):
                mask = mask[..., :seq_len, :seq_len]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = Attention.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_v)

        return torch.matmul(out, o_proj_weight.T)

    def forward(self):
        pass
