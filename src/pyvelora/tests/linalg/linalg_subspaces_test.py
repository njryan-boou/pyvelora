"""Tests for pyvelora/linalg/subspaces.py."""
import numpy as np
import pytest
from pyvelora.core import Matrix, Vector
from pyvelora.linalg.subspaces import (
    ColumnSpace, RowSpace, NullSpace, LeftNullSpace,
    column_space, row_space, null_space, left_null_space,
)


# Full-rank 3x3
A_full = Matrix([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]])

# Rank-2 matrix (third row = sum of first two)
A_rank2 = Matrix([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [5.0, 7.0, 9.0]])

# 2x3 matrix with one free variable
A_wide = Matrix([[1.0, 2.0, 1.0],
                 [0.0, 1.0, 1.0]])


# ---------------------------------------------------------------------------
# ColumnSpace
# ---------------------------------------------------------------------------

def test_column_space_full_rank_returns_3_vectors():
    cs = ColumnSpace(A_full)
    assert len(cs) == 3
    assert all(isinstance(v, Vector) for v in cs)


def test_column_space_rank2_returns_2_vectors():
    cs = ColumnSpace(A_rank2)
    assert len(cs) == 2


def test_column_space_wide_returns_2_vectors():
    cs = ColumnSpace(A_wide)
    assert len(cs) == 2


# ---------------------------------------------------------------------------
# RowSpace
# ---------------------------------------------------------------------------

def test_row_space_returns_rows_as_vectors():
    rs = RowSpace(A_rank2)
    assert len(rs) == 3
    assert all(isinstance(v, Vector) for v in rs)


def test_row_space_vectors_match_rows():
    rs = RowSpace(A_full)
    assert rs[0].data == [1.0, 0.0, 0.0]
    assert rs[1].data == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# NullSpace
# ---------------------------------------------------------------------------

def test_null_space_full_rank_is_empty():
    ns = NullSpace(A_full)
    assert ns == []


def test_null_space_rank2_returns_1_vector():
    ns = NullSpace(A_rank2)
    assert len(ns) == 1
    assert isinstance(ns[0], Vector)


def test_null_space_vector_satisfies_Ax_equals_zero():
    ns = NullSpace(A_rank2)
    for v in ns:
        result = [
            sum(A_rank2.data[r][c] * v.data[c] for c in range(A_rank2.shape[1]))
            for r in range(A_rank2.shape[0])
        ]
        assert all(abs(x) < 1e-10 for x in result)


def test_null_space_wide_returns_1_vector():
    ns = NullSpace(A_wide)
    assert len(ns) == 1


# ---------------------------------------------------------------------------
# LeftNullSpace
# ---------------------------------------------------------------------------

def test_left_null_space_full_rank_is_empty():
    lns = LeftNullSpace(A_full)
    assert lns == []


def test_left_null_space_rank2_returns_1_vector():
    lns = LeftNullSpace(A_rank2)
    assert len(lns) == 1


def test_left_null_space_vector_satisfies_ATx_equals_zero():
    lns = LeftNullSpace(A_rank2)
    AT = [[A_rank2.data[r][c] for r in range(A_rank2.shape[0])] for c in range(A_rank2.shape[1])]
    for v in lns:
        result = [
            sum(AT[r][c] * v.data[c] for c in range(len(AT[0])))
            for r in range(len(AT))
        ]
        assert all(abs(x) < 1e-10 for x in result)


# ---------------------------------------------------------------------------
# Lowercase aliases
# ---------------------------------------------------------------------------

def test_lowercase_column_space_alias():
    assert column_space is ColumnSpace


def test_lowercase_row_space_alias():
    assert row_space is RowSpace


def test_lowercase_null_space_alias():
    assert null_space is NullSpace


def test_lowercase_left_null_space_alias():
    assert left_null_space is LeftNullSpace


# ---------------------------------------------------------------------------
# Import via linalg namespace
# ---------------------------------------------------------------------------

def test_import_via_linalg():
    from pyvelora.linalg import column_space, row_space, null_space, left_null_space
    assert callable(column_space)
    assert callable(row_space)
    assert callable(null_space)
    assert callable(left_null_space)
