from . import loss
import torch

logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

# 0, 1
targets = torch.tensor([
    [1, 0, 0],
    [0, 1, 0]
])

loss_fn = loss.CrossEntropyLoss()
print(loss_fn(logits, targets))