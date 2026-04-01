import torch
import torch.nn as nn
import math

class LinearLayer(nn.Module):
    def __init__(self, input_features=2, output_features=3):
        super().__init__()
        self.W = nn.Parameter(torch.rand(output_features, input_features))
        self.bias = nn.Parameter(torch.rand(output_features))

    def forward(self, x):
        return x @ self.W.T + self.bias
    
class LinearLayerKaimingHe(nn.Module):
    def __init__(self, input_features=2, output_features=3, weights=None):
        super().__init__()
        if weights is None:
            # normal distribution with mean=0 and std=sqrt(2/input_features)
            W = torch.randn(output_features, input_features) * math.sqrt(2 / input_features)
        else:
            W = weights
        self.W = nn.Parameter(W)
        self.bias = nn.Parameter(torch.zeros(output_features, dtype=self.W.dtype, device=self.W.device), requires_grad=False)

    def forward(self, x):
        return x @ self.W.T + self.bias

if __name__ == "__main__":
    linear_layer = LinearLayer(input_features=2, output_features=3)
    print(linear_layer.W)
    print(linear_layer(torch.tensor([1.0, 2.0])))
