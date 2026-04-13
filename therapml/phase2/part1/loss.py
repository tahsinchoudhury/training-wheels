import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self, target_type: str = "one_hot") -> None:
        super().__init__()

        valid = {"indices", "one_hot"}
        if target_type not in valid:
            raise ValueError(f"target_type must be one of {valid}, got {target_type!r}")

        self.target_type = target_type

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets:
                - if target_type == "indices": Tensor of shape (batch_size,)
                  containing class ids
                - if target_type == "one_hot": Tensor of shape
                  (batch_size, num_classes) containing one-hot labels

        Returns:
            Scalar mean cross-entropy loss
        """

        m = torch.max(logits, dim=-1, keepdim=True).values
        
        # denominator - trick
        log_sum_exp = m + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True))
        
        # numerator = log(e^logit) = logit
        log_softmax = logits - log_sum_exp

        if "one_hot" == self.target_type:
            nll = -(log_softmax * targets).sum(dim=-1)
        else:
            batch_size = targets.shape[0]
            nll = -log_softmax[
                torch.arange(batch_size, device=logits.device),
                targets
            ]
        
        # average loss across all batches
        loss = nll.mean()

        return loss