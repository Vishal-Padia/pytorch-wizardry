import torch
from functions.swish import swish

if __name__ == "__main__":
    x = torch.randn(5, requires_grad=True)
    y = swish(x).sum()
    y.backward()

    print("x:", x)
    print("y:", y)
    print("x.grad:", x.grad)
