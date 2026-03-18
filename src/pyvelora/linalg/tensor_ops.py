from __future__ import annotations

import numpy as np

from ..core import Tensor

def contract(self, axes) -> Tensor:
    """Contract the tensor along specified axes."""
    return Tensor(np.tensordot(self.data, self.data, axes=axes))

def transpose(self, axes=None) -> Tensor:
    """Return the transposed tensor."""
    return Tensor(np.transpose(self.data, axes=axes))

def einsum(self, subscripts, *operands) -> Tensor:
    """Perform Einstein summation on the tensor."""
    return Tensor(np.einsum(subscripts, self.data, *operands))

def outer(self, other: Tensor) -> Tensor:
    """Return the outer product of this tensor with another."""
    return Tensor(np.outer(self.data, other.data))

def inner(self, other: Tensor) -> Tensor:
    """Return the inner product of this tensor with another."""
    return Tensor(np.inner(self.data, other.data))

def tensordot(self, other: Tensor, axes) -> Tensor:
    """Return the tensordot of this tensor with another along specified axes."""
    return Tensor(np.tensordot(self.data, other.data, axes=axes))

def kron(self, other: Tensor) -> Tensor:
    """Return the Kronecker product of this tensor with another."""
    return Tensor(np.kron(self.data, other.data))
