"""The numerical stability correction in calculateTransitionProbabilitiesBayflux.

When several proposals are far more likely than the current state, every
off-diagonal acceptance probability saturates at 1.0 and the row sum exceeds
one, driving the diagonal negative. The rescaling branch that repairs this was
never exercised.
"""

import numpy as np
import pytest

from bayflux.core.transitionProbabilities import (
    calculateTransitionProbabilitiesBayflux,
)


def transitionMatrixFor(likelihoods):
    numStates = len(likelihoods)
    samplesTemp = np.zeros((numStates, 3))
    return calculateTransitionProbabilitiesBayflux(
        samplesTemp, np.asarray(likelihoods, dtype=np.float64), numStates - 1
    )


def testRowsRemainStochasticWhenTwoProposalsSaturate():
    # states 1 and 2 are overwhelmingly better than state 0, so both
    # off-diagonal entries of row 0 saturate at 1.0 and sum to 2.0
    matrix = transitionMatrixFor([-1000.0, 0.0, 0.0])

    assert matrix.shape == (3, 3)
    for row in range(3):
        assert np.isclose(matrix[row].sum(), 1.0, atol=1e-9)
    assert (matrix >= 0.0).all()
    assert (matrix <= 1.0).all()


def testSaturatedRowIsRescaledRatherThanLeftNegative():
    matrix = transitionMatrixFor([-1000.0, 0.0, 0.0])

    # the current state cannot have a negative probability of staying put
    assert matrix[0, 0] >= 0.0
    # and the two equally good proposals must be reached with equal probability
    assert np.isclose(matrix[0, 1], matrix[0, 2])


def testManySaturatedProposalsStillGiveAStochasticMatrix():
    likelihoods = [-1000.0] + [0.0] * 8
    matrix = transitionMatrixFor(likelihoods)

    assert matrix.shape == (9, 9)
    for row in range(9):
        assert np.isclose(matrix[row].sum(), 1.0, atol=1e-9)
    assert (matrix >= 0.0).all()


def testEquallyLikelyStatesGiveASymmetricMatrix():
    matrix = transitionMatrixFor([-5.0, -5.0, -5.0])
    assert np.allclose(matrix, matrix.T)
