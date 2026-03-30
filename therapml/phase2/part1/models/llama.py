import torch
import torch.nn as nn

from ..transformer import TransformerBlock
from ..normalization import RMSNorm

class Llama(nn.Module):
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__()

        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict or None, got {type(config).__name__}")

        cfg = dict(config)
        cfg.update(kwargs)

        vocab_size = int(cfg.pop("vocab_size"))
        context_length = int(cfg.pop("context_length"))
        d_model = int(cfg.pop("d_model"))
        num_layers = int(cfg.pop("num_layers"))
        num_heads = int(cfg.pop("num_heads"))
        d_ff = int(cfg.pop("d_ff"))
        rope_theta = float(cfg.pop("rope_theta"))

        weights: dict[str, torch.Tensor] | None = cfg.pop("weights", None)
        tie_weights: bool = bool(cfg.pop("tie_weights", False))

        remove_rmsnorm: bool = bool(cfg.pop("remove_rmsnorm", False))
        use_post_norm: bool = bool(cfg.pop("use_post_norm", False))
        remove_rope: bool = bool(cfg.pop("remove_rope", False))
        ffn_type = cfg.pop("ffn_type", None)

        if cfg:
            raise TypeError(f"Unexpected config keys: {sorted(cfg.keys())}")

        if remove_rmsnorm:
            raise ValueError("remove_rmsnorm=True is not supported by this implementation")
        if use_post_norm:
            raise ValueError("use_post_norm=True is not supported by this implementation")
        if remove_rope:
            raise ValueError("remove_rope=True is not supported by this implementation")
        if ffn_type is not None:
            raise ValueError(f"ffn_type={ffn_type!r} is not supported by this implementation")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.tie_weights = tie_weights

        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    ctx_len=context_length,
                    theta=rope_theta,
                    weights=self._extract_layer_weights(weights, layer_idx) if weights is not None else None,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(hidden_dim=d_model, eps=1e-5)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

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
