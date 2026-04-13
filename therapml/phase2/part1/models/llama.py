from dataclasses import dataclass
from pathlib import Path
import json

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

    def save(self, path: str | Path) -> None:
        """Save model weights and config to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config as JSON
        config_data = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "tie_weights": self.tie_weights,
        }
        config_path = path.parent / f"{path.stem}_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Save model state dict
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "Llama":
        """Load model weights and config from disk."""
        path = Path(path)
        
        # Load config
        config_path = path.parent / f"{path.stem}_config.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        tie_weights = config_data.pop("tie_weights")
        config = LlamaConfig(**config_data)
        
        # Create model and load weights
        model = cls(config, tie_weights=tie_weights)
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)
        
        return model

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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        device: torch.device,
        max_new_tokens: int = 40000,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        full_ids = input_ids.tolist()[0]
        context_len = self.context_length

        for _ in range(int(max_new_tokens)):
            # Ensure input_ids has the correct shape [batch_size, seq_len]
            # and only contains the last `context_len` tokens.
            current_input_ids = torch.tensor([full_ids[-context_len:]], dtype=torch.long, device=device)

            logits = self.forward(current_input_ids)
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            full_ids.append(next_id)

            if eos_id is not None and next_id == int(eos_id):
                print("EOS token generated, stopping generation.")
                break

        return torch.tensor([full_ids], dtype=torch.long, device=device)

    @torch.no_grad()
    def num_parameters(self) -> tuple[int, int]:
        total_parameters = sum(p.numel() for p in self.parameters())
        total_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_parameters, total_trainable_parameters