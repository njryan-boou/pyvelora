from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
if TYPE_CHECKING:
    from pyvelora.core.vector.vector import Vector
# Note: The actual implementation of these functions would be integrated into the Vector class methods (e.g., __add__, __sub__, etc.) to provide operator overloading. The above functions are defined separately for clarity and modularity, but in practice, they would be called within the corresponding magic methods of the Vector class.

class VectorIndexing:
    """Vector indexing operations mixin."""

    def __iter__(self) -> Iterator[float | complex]:
        """Iterator for vector elements."""
        return iter(self.data)

    def __getitem__(self, index: int) -> float | complex:
        """Get item from vector at specified index."""
        return self.data[index]

    def __setitem__(self, index: int, value: float | complex) -> None:
        """Set item in vector at specified index."""
        self.data[index] = value

    def __contains__(self, item: float | complex) -> bool:
        """Check if item is in vector."""
        return any(a == item for a in self.data)

    def __len__(self) -> int:
        """Return the length of the vector."""
        return len(self.data)

