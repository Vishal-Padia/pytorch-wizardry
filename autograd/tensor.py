import numpy as np

class Tensor:
    """Core tensor class that wraps numpy arrays and enables automatic differentiation.
    Tracks the computational graph and handles gradient computation."""
    
    def __init__(self, data, requires_grad=False):
        # Convert input to numpy array with float32 dtype for consistency
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
            
        self.data = data  # Actual tensor data
        self.requires_grad = requires_grad  # Whether to compute gradients for this tensor
        self.grad = None  # Stores gradients once computed
        self._backward = lambda: None  # Function to compute gradients (set by Function.apply)
        self._prev = set()  # Set of immediate parent tensors in computational graph

    def backward(self, grad=None):
        """Computes gradients of the computational graph starting from this tensor.
        
        Args:
            grad: External gradient to backpropagate. For scalar outputs,
                 defaults to np.ones_like(data) if not specified."""
        
        if not self.requires_grad:
            return

        # For scalar outputs (size=1), use gradient of 1 if not specified
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data, dtype=np.float32)

        self.grad = grad

        # Build list of all tensors in computational graph in topological order
        # This ensures we compute gradients in correct order (leaf tensors last)
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:  # Visit all parent tensors first
                    build(child)
                topo.append(v)

        build(self)

        # Backpropagate gradients through the graph in reverse order
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"