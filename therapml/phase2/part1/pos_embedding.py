import torch
from torch import Tensor
from jaxtyping import Float, Int

class RoPE:

    def __init__(self, embedding_dim: int, theta: float, context_len: int):
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.theta = theta
        self.context_len = context_len
        self.half_dim = embedding_dim // 2

        # Shape: [half_dim]
        # inv_freq[i] = theta^(-2i / embedding_dim)
        dim_indices = torch.arange(0, self.half_dim, dtype=torch.float32)
        self.inv_freq = 1.0 / (theta ** (2.0 * dim_indices / embedding_dim))

        # Cache placeholders
        self._cached_cos = None
        self._cached_sin = None
        self._cached_device = None
        self._cached_dtype = None
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Build cos/sin cache up to seq_len.
        Cache shapes: [seq_len, half_dim]
        """
        need_rebuild = (
            self._cached_cos is None
            or self._cached_sin is None
            or self._cached_len < seq_len
            or self._cached_device != device
            or self._cached_dtype != dtype
        )

        if not need_rebuild:
            return

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # [seq_len]
        inv_freq = self.inv_freq.to(device=device)  # [half_dim]

        # angles[p, i] = position[p] * inv_freq[i]
        angles = torch.outer(positions, inv_freq)  # [seq_len, half_dim]

        self._cached_cos = angles.cos().to(dtype=dtype)
        self._cached_sin = angles.sin().to(dtype=dtype)
        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """
        For pairs (x1, x2), returns (-x2, x1).
        Input shape: [..., embedding_dim]
        Output shape: same
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Stack rotated pairs then flatten back
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def __call__(
        self,
        x: Float[Tensor, "batch ctx_len embedding_dim"],
        token_positions: Int[Tensor, "batch ctx_len"],
    ) -> Float[Tensor, "batch ctx_len embedding_dim"]:
        """
        Apply RoPE to x using provided token positions.

        Args:
            x: shape [batch, ctx_len, embedding_dim]
            token_positions: shape [batch, ctx_len]

        Returns:
            Tensor of shape [batch, ctx_len, embedding_dim]
        """
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Last dimension of x must be {self.embedding_dim}, got {x.shape[-1]}"
            )

        if token_positions.ndim == 1:
            if token_positions.shape[0] != x.shape[1]:
                raise ValueError(
                    f"1D token_positions must have length {x.shape[1]}, got {token_positions.shape[0]}"
                )
            token_positions = token_positions.unsqueeze(0).expand(x.shape[0], -1)

        if token_positions.shape != x.shape[:2]:
            raise ValueError(
                f"token_positions shape must match x[:2]. "
                f"Got token_positions={token_positions.shape}, x[:2]={x.shape[:2]}"
            )

        max_pos = int(token_positions.max().item()) + 1
        if max_pos > self.context_len:
            raise ValueError(
                f"token_positions contains position {max_pos - 1}, "
                f"but context_len is only {self.context_len}"
            )

        self._build_cache(
            seq_len=self.context_len,
            device=x.device,
            dtype=x.dtype,
        )

        # Gather cos/sin for each token position
        # token_positions: [batch, ctx_len]
        # cos/sin after indexing: [batch, ctx_len, half_dim]
        cos = self._cached_cos[token_positions]
        sin = self._cached_sin[token_positions]

        # Expand each frequency value to cover both members of each rotated pair
        # [batch, ctx_len, half_dim] -> [batch, ctx_len, embedding_dim]
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)

        return x * cos + self._rotate_half(x) * sin
