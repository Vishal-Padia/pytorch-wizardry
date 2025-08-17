import torch
from torch.autograd import gradcheck
from functions.swish import swish


def test_swish_grad():
    """
    Use gradcheck to verify correctness of backward.
    Hint: gradcheck needs double precision (dtype=torch.float64).
    """
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(swish, (x,), eps=1e-6, atol=1e-4)
