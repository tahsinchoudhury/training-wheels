import torch
import torch.nn as nn
import math
from jaxtyping import Float


class Sigmoid(nn.Module):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class SiLU(nn.Module):
    def forward(self, x):
        sigmoid = Sigmoid()
        return x * sigmoid(x)

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x > 0, x, torch.tensor(0))
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return 0.5 * x * (1 + torch.tanh(
        #     math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        # ))
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, w1_weight, w2_weight, w3_weight):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = nn.Parameter(w1_weight)
        self.w2_weight = nn.Parameter(w2_weight)
        self.w3_weight = nn.Parameter(w3_weight)

        self.silu = nn.SiLU()

    def forward(self, x) -> Float[torch.Tensor, "... d_model"]:
        x1 = x @ self.w1_weight.T
        x2 = x @ self.w3_weight.T

        swiglu = x2 * self.silu(x1)

        ret = swiglu @ self.w2_weight.T
        return ret
    
class Softmax(nn.Module):
    def forward(self, x, dim=1):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    
if __name__ == "__main__":
    x = torch.tensor([1, 2])
    
    # relu = ReLU()
    # print(relu(x))

    # gelu = GELU()
    # print(gelu(x))
    softmax = Softmax()
    print(softmax(x.unsqueeze(0)))