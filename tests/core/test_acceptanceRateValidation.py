"""Validation and conversion paths of bayflux.core.acceptanceRate.

test_acceptanceRate.py covers the arithmetic. This file covers the input
handling branches it does not reach: array-like objects that are neither a
list, a tuple nor an ndarray, and the several ways conversion to a numeric
array can fail.
"""

import numpy as np
import pandas as pd
import pytest

from bayflux.core.acceptanceRate import computeAcceptanceRate


def testPandasSeriesIsAccepted():
    samples = pd.Series([1, 0, 1, 0, 1, 1])
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 4 / 6)


def testEmptyPandasSeriesIsRejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        computeAcceptanceRate(pd.Series([], dtype=float), burnIn=0)


def testObjectWithoutLengthIsRejected():
    with pytest.raises(TypeError, match="non-empty array-like object"):
        computeAcceptanceRate(42, burnIn=0)


def testEmptyNumpyArrayIsRejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        computeAcceptanceRate(np.array([]), burnIn=0)


def testEmptyTupleIsRejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        computeAcceptanceRate((), burnIn=0)


def testEmptySubArrayIsRejected():
    with pytest.raises(ValueError, match="empty sub-arrays"):
        computeAcceptanceRate([np.array([1, 0]), np.array([])], burnIn=0)


def testZeroDimensionalSubArraysAreRejected():
    with pytest.raises(ValueError, match="zero-dimensional"):
        computeAcceptanceRate([np.array(1.0), np.array(0.0)], burnIn=0)


def testNonNumericValuesAreRejected():
    with pytest.raises(ValueError, match="Failed to convert input"):
        computeAcceptanceRate([[1, 0], ["accepted", 0]], burnIn=0)


def testUnconvertibleObjectsAreRejected():
    with pytest.raises(TypeError, match="cannot be converted to numeric"):
        computeAcceptanceRate([{"accepted": 1}, {"accepted": 0}], burnIn=0)


def testRaggedSubArraysStillConcatenate():
    # concatenation flattens chains of unequal length, which is intended:
    # MPI tasks are not required to contribute the same number of proposals
    rate = computeAcceptanceRate([np.array([1, 1, 1]), np.array([0])], burnIn=0)
    assert np.isclose(rate, 3 / 4)
