import torch
import pytest
from .adapters import get_sgd_cls, get_adam_cls


def _optimize(opt_class, cfg) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(model.parameters(), **cfg)

    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()

    return model.weight.detach()


def test_sgd():
    cfg = dict(
        lr=1e-3,
        weight_decay=0.01,
    )

    pytorch_weights = _optimize(torch.optim.SGD, cfg)
    actual_weights = _optimize(get_sgd_cls(), cfg)

    assert torch.allclose(actual_weights, pytorch_weights, atol=1e-4)

# @pytest.mark.skip(reason="Not implemented yet")
def test_adam():
    cfg = dict(
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    pytorch_weights = _optimize(torch.optim.AdamW, cfg)
    actual_weights = _optimize(get_adam_cls(), cfg)

    assert torch.allclose(pytorch_weights, actual_weights, atol=1e-4)
