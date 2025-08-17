from tensor import Tensor
from ops import Add, Mul

# Create input tensors with gradient tracking enabled
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

z = Add.apply(Mul.apply(x, y), y)

# Compute gradients via backpropagation
z.backward()

# Display results
print(f"z : {z}")
print(f"x.grad: {x.grad}")  # x = 3
print(f"y.grad: {y.grad}")  # y = 2 + 1 = 3
