from dataclasses import dataclass

import torch
import torch.nn as nn

from ..transformer import TransformerBlock
from ..normalization import RMSNorm


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab_size", int(self.vocab_size))
        object.__setattr__(self, "context_length", int(self.context_length))
        object.__setattr__(self, "d_model", int(self.d_model))
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "d_ff", int(self.d_ff))
        object.__setattr__(self, "rope_theta", float(self.rope_theta))


class Llama(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        weights: dict[str, torch.Tensor] | None = None,
        tie_weights: bool = False,
    ):
        super().__init__()

        if not isinstance(config, LlamaConfig):
            raise TypeError(f"config must be a LlamaConfig instance, got {type(config).__name__}")

        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.rope_theta = config.rope_theta
        self.tie_weights = tie_weights

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    ctx_len=config.context_length,
                    theta=config.rope_theta,
                    weights=self._extract_layer_weights(weights, layer_idx) if weights is not None else None,
                )
                for layer_idx in range(config.num_layers)
            ]
        )

        self.ln_final = RMSNorm(hidden_dim=config.d_model, eps=1e-5)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if tie_weights:
            self._tie_embedding_and_head()

        if weights is not None:
            self._load_from_weights_dict(weights)

    @staticmethod
    def _extract_layer_weights(
        weights: dict[str, torch.Tensor] | None,
        layer_idx: int,
    ) -> dict[str, torch.Tensor] | None:
        if weights is None:
            return None
        prefix = f"layers.{layer_idx}."
        layer_weights = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        return layer_weights

    def _tie_embedding_and_head(self) -> None:
        if self.lm_head.weight.shape != self.token_embeddings.weight.shape:
            raise ValueError(
                f"Cannot tie weights with shapes lm_head={tuple(self.lm_head.weight.shape)} "
                f"and token_embeddings={tuple(self.token_embeddings.weight.shape)}"
            )
        self.lm_head.weight = self.token_embeddings.weight

    def _load_from_weights_dict(self, weights: dict[str, torch.Tensor]) -> None:
        required = [
            "token_embeddings.weight",
            "ln_final.weight",
            "lm_head.weight",
        ]
        missing = [k for k in required if k not in weights]
        if missing:
            raise KeyError(f"Missing required weights: {missing}")

        if self.tie_weights and not torch.allclose(weights["token_embeddings.weight"], weights["lm_head.weight"]):
            raise ValueError(
                "tie_weights=True requires token_embeddings.weight and lm_head.weight to be identical. "
                "Disable tie_weights for this weight set."
            )

        with torch.no_grad():
            self.token_embeddings.weight.copy_(weights["token_embeddings.weight"])
            self.ln_final.gamma.copy_(weights["ln_final.weight"])

            self.lm_head.weight.copy_(weights["lm_head.weight"])

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        if in_indices.ndim != 2:
            raise ValueError(f"Expected in_indices to have shape [batch, seq_len], got {tuple(in_indices.shape)}")

        in_indices = in_indices.to(dtype=torch.long)
        batch, seq_len = in_indices.shape
        if seq_len > self.context_length:
            raise ValueError(f"sequence_length ({seq_len}) must be <= context_length ({self.context_length})")

        x = self.token_embeddings(in_indices)

        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0)

        for block in self.layers:
            x = block(x, token_positions=token_positions, mask=mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
