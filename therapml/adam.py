# import torch

# class AdamW(torch.optim.Optimizer):
#     def __init__(self, params, lr=0.005, weight_decay=0.01):
#         defaults = {
#             "lr": lr,
#             "weight_decay": weight_decay,
#         }
#         super().__init__(params, defaults)

#     def step(self, closure=None):
#         assert closure is None
#         loss = 0.0

#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 p.data -= group["lr"] * p.grad.data + group["lr"] * group["weight_decay"] * p.data

#         return loss