from function import Function


class Add(Function):
    """Addition operation implementing z = a + b

    Forward: z = a + b
    Backward: dz/da = 1, dz/db = 1

    The gradient of addition with respect to both inputs is 1,
    so we just pass the incoming gradient unchanged to both."""

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dz where L is the final scalar loss
        # For addition: dL/da = dL/dz * dz/da = dL/dz * 1
        return grad_output, grad_output


class Mul(Function):
    """Multiplication operation implementing z = a * b

    Forward: z = a * b
    Backward: dz/da = b, dz/db = a (product rule)

    The gradient of multiplication follows the product rule:
    - Gradient wrt a: dz/da = b
    - Gradient wrt b: dz/db = a"""

    @staticmethod
    def forward(ctx, a, b):
        # Need to save inputs for product rule in backward pass
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # grad_output is dL/dz
        # For multiplication:
        # dL/da = dL/dz * dz/da = dL/dz * b
        # dL/db = dL/dz * dz/db = dL/dz * a
        return grad_output * b, grad_output * a
