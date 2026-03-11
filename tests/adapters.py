from typing import Any
from jaxtyping import Float, Int, Bool

from torch import Tensor
from therapml import tensor
from therapml import sgd
from therapml.phase2.part1.linear_layer import LinearLayerKaimingHe
from therapml.phase2.part1 import activations, loss, optimizers, dropout, normalization


def run_tensor_multiply(arr1: Float[list, "b x y"], arr2: Float[list, "b y z"]) -> Float[list, "b x z"]:
    # raise NotImplementedError
    return tensor.MatrixOps.tensor_multiply(arr1, arr2)


def run_tensor_dot(arr1: Float[list, "..."], arr2: Float[list, "..."], dim: int):
    # raise NotImplementedError
    return tensor.MatrixOps.tensor_dot(arr1, arr2, dim)


def get_sgd_cls() -> Any:
    # raise NotImplementedError
    return sgd.SGD

def get_adam_cls() -> Any:
    return optimizers.AdamW

def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # raise NotImplementedError
    relu = activations.ReLU()
    return relu(in_features)


def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    gelu = activations.GELU()
    return gelu(in_features)


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    softmax = activations.Softmax()
    return softmax(x=in_features, dim=dim)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    # raise NotImplementedError
    linear_layer = LinearLayerKaimingHe(
        input_features=d_in,
        output_features=d_out,
        weights=weights
    )
    return linear_layer(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # raise NotImplementedError
    swiglu = activations.SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
    )
    return swiglu(in_features)


def run_cross_entropy_loss(
    logits: Float[Tensor, "batch output_dim"], ground_truth: Float[Tensor, "batch output_dim"]
) -> Float[Tensor, ""]:
    # raise NotImplementedError
    loss_fn = loss.CrossEntropyLoss()
    return loss_fn(logits, ground_truth)


def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    # raise NotImplementedError
    dropout_layer = dropout.Dropout(prob)
    return dropout_layer(input)


def run_layernorm(
    input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."], beta: Float[Tensor, "batch ..."]
) -> Float[Tensor, "batch ..."]:
    hidden_dim = input.shape[-1]
    layernorm = normalization.LayerNorm(
        hidden_dim=hidden_dim,
        gamma=gamma,
        beta=beta
    )
    return layernorm(input)


def run_rmsnorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    hidden_dim = input.shape[-1]
    rmsnorm = normalization.RMSNorm(
        hidden_dim=hidden_dim,
        gamma=gamma
    )
    return rmsnorm(input)


def run_rope(
    embedding_dim: int,
    theta: float,
    context_len: int,
    input_embeddings: Float[Tensor, "batch ctx_len embedding_dim"],
    token_positions: Int[Tensor, "batch ctx_len"],
) -> Float[Tensor, "batch ctx_len embedding_dim"]:
    raise NotImplementedError


def run_self_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... batch keys d_k"],
    V: Float[Tensor, "... batch values d_v"],
    mask: Bool[Tensor, "... batch queries keys"] | None = None,
) -> Float[Tensor, "... batch queries d_v"]:
    """
    Note the number of dimensions here can be greater than 3
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "batch ctx_len d_in"],
) -> Float[Tensor, "batch ctx_len d_out"]:
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
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
) -> Float[Tensor, "batch ctx_len d_out"]:
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    ctx_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Note this function should use RoPE.

    Args:
        d_model (int): The dimensionality of the Transformer block input.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer.

        ctx_len (int): Maximum sequence length of the input_tensor.

        theta (float): RoPE parameter.

        weights (dict[str, Tensor]):
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).

        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Similar to before it uses RoPE
    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.

        context_length (int): The maximum number of tokens to process at once.

        d_model (int): The dimensionality of the model embeddings and sublayer outputs.

        num_layers (int): The number of Transformer layers to use.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer (See Vaswani et al).

        rope_theta (float): The RoPE `theta` parameter.

        weights (dict[str, Tensor]):
            {num_layers} refers to an integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError
