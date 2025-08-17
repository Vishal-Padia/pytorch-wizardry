import numpy as np
from tensor import Tensor

class Context:
    """Stores intermediate values needed for gradient computation.
    Each operation (Function) gets its own context during forward pass."""
    
    def __init__(self):
        self.saved_tensors = ()  # Tuple of tensors needed for backward pass

    def save_for_backward(self, *tensors):
        """Save tensors that will be needed during backward pass.
        Example: Mul operation saves both inputs to compute gradients later."""
        self.saved_tensors = tensors

class Function:
    """Base class for all differentiable operations.
    Handles the machinery of forward/backward passes and gradient computation."""

    @classmethod
    def apply(cls, *tensors):
        """Executes the operation and sets up gradient computation.
        
        Args:
            tensors: Input tensors for the operation
            
        Returns:
            A new tensor with _backward function set up for gradient computation"""
        
        # Create context for storing intermediate values
        ctx = Context()
        
        # Extract raw numpy arrays from inputs, convert constants to arrays
        raw_inputs = [t.data if isinstance(t, Tensor) else np.array(t, dtype=np.float32) for t in tensors]
        
        # Compute forward pass
        out_data = cls.forward(ctx, *raw_inputs)

        # Ensure output is numpy array
        if not isinstance(out_data, np.ndarray):
            out_data = np.array(out_data, dtype=np.float32)

        # Output requires gradients if any input requires gradients
        requires_grad = any(t.requires_grad for t in tensors)
        
        # Create output tensor and connect to inputs in graph
        out = Tensor(out_data, requires_grad=requires_grad)
        out._prev = set(tensors)

        def _backward():
            """Computes gradients of operation with respect to inputs.
            This is called during backward() traversal of computational graph."""
            
            grads = cls.backward(ctx, out.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)  # Handle single gradient case

            # Distribute gradients to input tensors
            for t, g in zip(tensors, grads):
                if isinstance(t, Tensor) and t.requires_grad:
                    # Accumulate gradients (handles case where tensor is used multiple times)
                    if t.grad is None:
                        t.grad = g
                    else:
                        t.grad = t.grad + g

        # Only set up backward function if gradients are needed
        if requires_grad:
            out._backward = _backward

        return out