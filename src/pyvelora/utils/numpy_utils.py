from __future__ import annotations
import numpy as np

def linspace(start: float, stop: float, num: int) -> np.ndarray:
    """Return evenly spaced numbers over a specified interval."""
    return np.linspace(start, stop, num)