import torch
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SwishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass for Swish:
            swish(x) = x * sigmoid(x)

        - save tensors using ctx.save_for_backward(...)
        - return the forward output as a torch.Tensor
        """

        def sig(x):
            return 1 / (1 + np.exp(-x))

        sigmoid = sig(input)
        out = input * sigmoid
        ctx.save_for_backward(input, sigmoid)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass:
            d/dx [x * sigmoid(x)]
        Steps:
            - retrieve saved tensors with ctx.saved_tensors
            - compute gradient wrt input
            - return gradient(s) (must match number of inputs to forward)
        """
        input, sigmoid = ctx.saved_tensors
        grad_output = grad_output * (sigmoid + input * sigmoid * (1 - sigmoid))
        return (grad_output,)


# convenience wrapper so you can call swish(x) instead of SwishFn.apply(x)
def swish(x):
    return SwishFn.apply(x)
