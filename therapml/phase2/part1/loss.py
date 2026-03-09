import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, num_classes)
        # targets: (batch_size)

        m = torch.max(logits, dim=1, keepdim=True).values
        
        # denominator - trick
        log_sum_exp = m + torch.log(torch.sum(torch.exp(logits - m), dim=1, keepdim=True))
        
        # numerator = log(e^logit) = logit
        log_softmax = logits - log_sum_exp

        nll = -(log_softmax * targets).sum(dim=1)
        
        # average loss across all batches
        loss = nll.mean()

        return loss