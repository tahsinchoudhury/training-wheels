from pathlib import Path

import torch
from therapml.phase2.part1.tokenizer.bpe import BPETokenizer
from therapml.phase2.part1.loss import CrossEntropyLoss


def _load_tokenizer(tokenizer_path: Path) -> BPETokenizer | ValueError:
    if tokenizer_path.exists():
        return BPETokenizer.load(tokenizer_path)

    return ValueError(f"Tokenizer not found at {tokenizer_path}. Please provide a valid path to a trained tokenizer.")

tokenizer_path = Path("data/tinystories_bpe.json")
tokenizer = _load_tokenizer(tokenizer_path)
if isinstance(tokenizer, ValueError):
    print(tokenizer)
else:
    print(f"Loaded tokenizer vocab_size={tokenizer.vocab_size} from {tokenizer_path}")
    print(tokenizer.token_to_id("<eos>"))


def test_crossentropy_loss_target_type_equivalence():
    """
    Test that CrossEntropyLoss produces the same output when using
    target_type="one-hot" and target_type="indices" with properly formatted targets.
    """
    batch_size = 4
    num_classes = 10
    
    # Create dummy logits
    logits = torch.randn(batch_size, num_classes)
    
    # Create target indices (shape: (batch_size,))
    target_indices = torch.tensor([1, 3, 7, 2], dtype=torch.long)
    
    # Convert indices to one-hot (shape: (batch_size, num_classes))
    target_one_hot = torch.zeros(batch_size, num_classes)
    target_one_hot.scatter_(1, target_indices.unsqueeze(1), 1.0)
    
    # Compute loss with both target types
    loss_fn_indices = CrossEntropyLoss(target_type="indices")
    loss_fn_one_hot = CrossEntropyLoss(target_type="one_hot")
    
    loss_indices = loss_fn_indices(logits, target_indices)
    loss_one_hot = loss_fn_one_hot(logits, target_one_hot)
    
    # Check if losses are approximately equal
    tolerance = 1e-6
    assert torch.allclose(loss_indices, loss_one_hot, atol=tolerance), (
        f"Losses do not match! "
        f"loss_indices={loss_indices.item():.6f}, "
        f"loss_one_hot={loss_one_hot.item():.6f}, "
        f"diff={abs(loss_indices.item() - loss_one_hot.item()):.6e}"
    )
    
    print("✓ CrossEntropyLoss target_type equivalence test passed!")
    print(f"  Loss (indices): {loss_indices.item():.6f}")
    print(f"  Loss (one-hot): {loss_one_hot.item():.6f}")

test_crossentropy_loss_target_type_equivalence()