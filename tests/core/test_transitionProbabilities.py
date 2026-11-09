import numpy as np
import pytest
from bayflux.core.transitionProbabilities import calculateTransitionProbabilitiesBayflux


def testCalculateTransitionProbabilitiesBasic():
    """
    Test basic functionality of calculateTransitionProbabilitiesBayflux.
    
    Verifies that the function returns a properly sized stochastic matrix
    where each row sums to 1.
    """
    numProposals = 2
    samplesTemp = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])
    likelihoods = np.array([np.log(0.2), np.log(0.4), np.log(0.1)])
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, numProposals)
    assert A.shape == (3, 3)
    for i in range(3):
        assert np.isclose(np.sum(A[i]), 1.0, atol=1e-6)


def testInvalidInputLengths():
    """
    Test error handling when likelihoods and samples have mismatched lengths.
    
    Verifies that the function raises a ValueError when the number of likelihoods
    doesn't match the number of samples.
    """
    samplesTemp = np.array([[0.1, 0.2]])
    likelihoods = np.array([np.log(0.2), np.log(0.3)])  # too long
    with pytest.raises(ValueError, match="must have the same non-zero length"):
        calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, 1)


def testEmptyInputs():
    """
    Test error handling when empty inputs are provided.
    
    Verifies that the function raises a ValueError when empty arrays are passed.
    """
    samplesTemp = np.array([]).reshape(0, 2)
    likelihoods = np.array([])
    with pytest.raises(ValueError, match="must have the same non-zero length"):
        calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, 0)


def testEqualLikelihoods():
    """
    Test behavior when all likelihoods are equal.
    
    Verifies that the function produces valid transition probabilities
    when all samples have the same likelihood, which should result in
    uniform transition probabilities.
    """
    numProposals = 2
    samplesTemp = np.ones((3, 2))
    likelihoods = np.log(np.ones(3))  # All log(1) = 0
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, numProposals)
    for i in range(3):
        for j in range(3):
            assert 0 <= A[i, j] <= 1
        assert np.isclose(np.sum(A[i]), 1.0, atol=1e-6)


def testLargeLikelihoodDifferences():
    """
    Test behavior with extreme differences in log-likelihoods.
    
    Verifies that the function handles large numerical differences in
    log-likelihoods without producing NaN or infinity values, and still
    returns a valid stochastic matrix.
    """
    samplesTemp = np.ones((3, 2))
    likelihoods = np.array([-1000.0, 0.0, 1000.0])
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, 2)
    assert np.all(np.isfinite(A))
    for i in range(3):
        assert np.isclose(np.sum(A[i]), 1.0, atol=1e-6)


def testNegativeLogLikelihoods():
    """
    Test behavior with negative log-likelihoods.
    
    Verifies that the function correctly handles negative log-likelihood values,
    which are common when probabilities are less than 1.
    """
    samplesTemp = np.ones((3, 2))
    likelihoods = np.log(np.array([0.5, 0.1, 0.2]))  # logs are negative
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, 2)
    assert A.shape == (3, 3)
    for i in range(3):
        assert np.isclose(np.sum(A[i]), 1.0, atol=1e-6)


def testZeroProposals():
    """
    Test behavior when numProposals is set to zero.
    
    Verifies that when no proposals are requested, the function returns
    an identity matrix (each state only transitions to itself).
    """
    samplesTemp = np.array([[0.1, 0.2]])
    likelihoods = np.array([np.log(0.5)])
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, 0)
    assert A.shape == (1, 1)
    assert np.isclose(A[0, 0], 1.0)


def testProbabilityBounds():
    """
    Test that all transition probabilities are properly bounded.
    
    Verifies that all entries in the transition matrix are between 0 and 1,
    and that each row sums to 1, even with random inputs.
    """
    numProposals = 3
    samplesTemp = np.random.rand(numProposals + 1, 5)
    likelihoods = np.random.randn(numProposals + 1) * 10
    A = calculateTransitionProbabilitiesBayflux(samplesTemp, likelihoods, numProposals)
    assert np.all((A >= 0) & (A <= 1))
    for i in range(numProposals + 1):
        assert np.isclose(np.sum(A[i]), 1.0, atol=1e-6)