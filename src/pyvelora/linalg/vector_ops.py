from __future__ import annotations
import numpy as np

from pyvelora.core import Vector, Matrix

def dot(v1: Vector, v2: Vector) -> float:
    """Return the dot product of two vectors."""
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Dot product is only defined for Vector objects.")
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape for dot product.")
    result = np.sum([a * b for a, b in zip(v1.data, v2.data)])
    return result

def cross(v1: Vector, v2: Vector) -> Vector:
    """Return the cross product of two 3D vectors."""
    if not isinstance(v1, Vector) or not isinstance(v2, Vector):
        raise TypeError("Cross product is only defined for Vector objects.")
    if v1.shape != (3,) or v2.shape != (3,):
        raise ValueError("Cross product is only defined for 3D vectors.")
    
    x = v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1]
    y = v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2]
    z = v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0]
    return Vector([x, y, z])

def project(v1: Vector, v2: Vector) -> Vector:
    """Return the projection of v1 onto v2."""
    return (dot(v1, v2) / dot(v2, v2)) * v2

def angle_between(v1: Vector, v2: Vector, degrees=False) -> float:
    """Return the angle between two vectors in radians (or degrees if specified)."""
    cos_theta = dot(v1, v2) / (np.linalg.norm(v1.data) * np.linalg.norm(v2.data))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle) if degrees else angle

def triple_product(v1: Vector, v2: Vector, v3: Vector) -> float:
    """Return the scalar triple product of three vectors."""
    return dot(v1, cross(v2, v3))

def mixed_product(v1: Vector, v2: Vector, v3: Vector) -> float:
    """Return the mixed product of three vectors."""
    return dot(cross(v1, v2), v3)

def vector_norm(v: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    if not isinstance(v, Vector):
        raise TypeError("Norm is only defined for Vector objects.")
    return (dot(v, v)) ** 0.5

def normalize(v: Vector) -> Vector:
    """Return the unit vector in the direction of v."""
    norm = vector_norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return Vector(v.data / norm)

def angle_between_planes(n1: Vector, n2: Vector, degrees=False) -> float:
    """Return the angle between two planes defined by their normal vectors."""
    return angle_between(n1, n2, degrees=degrees)