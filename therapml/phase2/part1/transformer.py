import torch
import torch.nn as nn

from torch import Tensor

from .activations import SwiGLU
from .attention import MultiHeadAttention
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        ctx_len: int,
        theta: float,
        weights: dict[str, Tensor] | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ctx_len = ctx_len
        self.theta = theta

        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn.output_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = RMSNorm(hidden_dim=d_model, eps=1e-5)

        w1 = nn.Linear(d_model, d_ff, bias=False)
        w2 = nn.Linear(d_ff, d_model, bias=False)
        w3 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            w1_weight=w1.weight.detach().clone(),
            w2_weight=w2.weight.detach().clone(),
            w3_weight=w3.weight.detach().clone(),
        )

        self.ln2 = RMSNorm(hidden_dim=d_model, eps=1e-5)

        if weights is not None:
            self._load_from_weights_dict(weights)

    def _load_from_weights_dict(self, weights: dict[str, Tensor]) -> None:
        required = [
            "attn.q_proj.weight",
            "attn.k_proj.weight",
            "attn.v_proj.weight",
            "attn.output_proj.weight",
            "ln1.weight",
            "ffn.w1.weight",
            "ffn.w2.weight",
            "ffn.w3.weight",
            "ln2.weight",
        ]
        missing = [k for k in required if k not in weights]
        if missing:
            raise KeyError(f"Missing required weights: {missing}")

        with torch.no_grad():
            self.attn.q_proj.weight.copy_(weights["attn.q_proj.weight"])
            self.attn.k_proj.weight.copy_(weights["attn.k_proj.weight"])
            self.attn.v_proj.weight.copy_(weights["attn.v_proj.weight"])
            self.attn.output_proj.weight.copy_(weights["attn.output_proj.weight"])

            self.ln1.gamma.copy_(weights["ln1.weight"])
            self.ln2.gamma.copy_(weights["ln2.weight"])

            self.ffn.w1_weight.copy_(weights["ffn.w1.weight"])
            self.ffn.w2_weight.copy_(weights["ffn.w2.weight"])
            self.ffn.w3_weight.copy_(weights["ffn.w3.weight"])
            
    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Pre-norm transformer block with RoPE:
          x = x + MHA(RMSNorm(x))
          x = x + FFN(RMSNorm(x))
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [batch, seq_len, d_model], got {tuple(x.shape)}")
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")
        if seq_len > self.ctx_len:
            raise ValueError(f"seq_len ({seq_len}) must be <= ctx_len ({self.ctx_len})")

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)
            if token_positions.ndim == 1:
                token_positions = token_positions.unsqueeze(0).expand(batch, -1)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)

        residual = x
        x_norm = self.ln1(x)
        attn_out = MultiHeadAttention.multihead_attention_with_rope(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ctx_len=self.ctx_len,
            theta=self.theta,
            q_proj_weight=self.attn.q_proj.weight,
            k_proj_weight=self.attn.k_proj.weight,
            v_proj_weight=self.attn.v_proj.weight,
            o_proj_weight=self.attn.output_proj.weight,
            in_features=x_norm,
            token_positions=token_positions,
            mask=mask,
        )
        x = residual + attn_out

        residual = x
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        return x

class Transformer(nn.Module):
    pass 
