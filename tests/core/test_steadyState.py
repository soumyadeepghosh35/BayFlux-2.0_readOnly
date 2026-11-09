import numpy as np
import pytest
from bayflux.core.steadyState import steadyStateDistribution

class TestSteadyStateDistribution:
    """
    Comprehensive test suite for the steadyStateDistribution function.
    
    This test suite covers various scenarios including:
    1. Basic functionality with simple transition matrices
    2. Edge cases such as identity matrices and uniform transition matrices
    3. Error handling for invalid inputs
    4. Convergence behavior with different tolerance values
    5. Performance with larger matrices
    """

    def testBasicFunctionality(self):
        """
        Test basic functionality with a simple 2x2 transition matrix.
        
        This test verifies that the function correctly computes the steady-state
        distribution for a simple Markov chain.
        """
        # Simple 2x2 transition matrix
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        # Analytical solution: π = [4/7, 3/7]
        expected = np.array([4/7, 3/7])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Check if result matches expected values
        assert np.allclose(result, expected, rtol=1e-5)
        
        # Verify that the result is a probability distribution (sums to 1)
        assert np.isclose(np.sum(result), 1.0)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)

    def testIdentityMatrix(self):
        """
        Test with an identity matrix.
        
        For an identity matrix, every state is absorbing, and the steady-state
        distribution should be equal to the initial distribution (uniform in this case).
        """
        # Identity matrix (every state is absorbing)
        n = 4
        P = np.eye(n)
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # For an identity matrix, any probability distribution is a steady state
        # Since we start with a uniform distribution, we expect a uniform result
        expected = np.ones(n) / n
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testUniformTransitionMatrix(self):
        """
        Test with a uniform transition matrix.
        
        A uniform transition matrix has equal probability of transitioning to any state.
        The steady-state distribution should be uniform.
        """
        # Uniform transition matrix (equal probability to transition to any state)
        n = 5
        P = np.ones((n, n)) / n
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Expected: uniform distribution
        expected = np.ones(n) / n
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testCyclicMatrix(self):
        """
        Test with a cyclic transition matrix.
        
        A cyclic matrix represents a Markov chain where states are visited in a cycle.
        For a symmetric cyclic matrix, the steady-state distribution should be uniform.
        """
        # Cyclic matrix (state 0 -> 1 -> 2 -> 0)
        P = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Expected: uniform distribution for symmetric cycle
        expected = np.ones(3) / 3
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testWeightedCyclicMatrix(self):
        """
        Test with a weighted cyclic transition matrix.
    
        This test verifies that the function correctly handles asymmetric
        transition probabilities in a cyclic Markov chain.
        """
        # Weighted cyclic matrix
        P = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)
        
        # For this specific matrix, the correct analytical solution is [0.2, 0.4, 0.4]
        expected = np.array([0.2, 0.4, 0.4])
        assert np.allclose(result, expected, rtol=1e-5)
        

    def testAbsorbingState(self):
        """
        Test with a Markov chain containing an absorbing state.
        
        An absorbing state is one that, once entered, cannot be left.
        The steady-state distribution should have all probability mass on the absorbing state.
        """
        # Matrix with an absorbing state (state 0)
        P = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.3, 0.3, 0.4]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Expected: all probability mass on the absorbing state
        expected = np.array([1.0, 0.0, 0.0])
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testMultipleAbsorbingStates(self):
        """
        Test with a Markov chain containing multiple absorbing states.
        
        With multiple absorbing states, the steady-state distribution depends on
        the initial distribution and the structure of the transition matrix.
        """
        # Matrix with two absorbing states (states 0 and 2)
        P = np.array([
            [1.0, 0.0, 0.0],
            [0.4, 0.2, 0.4],
            [0.0, 0.0, 1.0]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)
        
        # Check that non-absorbing states have zero probability
        assert np.isclose(result[1], 0.0, atol=1e-5)
        
        # The distribution between absorbing states depends on the initial distribution
        # and transition structure, so we don't check specific values

    def testReducibleMatrix(self):
        """
        Test with a reducible transition matrix.
        
        A reducible Markov chain has multiple communicating classes.
        The steady-state distribution depends on the initial distribution.
        """
        # Reducible matrix with two communicating classes: {0,1} and {2,3}
        P = np.array([
            [0.7, 0.3, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.8, 0.2]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testCustomTolerance(self):
        """
        Test with a custom convergence tolerance.
        
        This test verifies that the function respects the provided tolerance parameter.
        """
        # Simple transition matrix
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        # Compute with different tolerances
        result_default = steadyStateDistribution(P)
        result_custom = steadyStateDistribution(P, tolerance=1e-5)
        
        # Both should converge to the same distribution
        assert np.allclose(result_default, result_custom, rtol=1e-5)

    def testCustomMaxIterations(self):
        """
        Test with a custom maximum iteration count.
        
        This test verifies that the function respects the provided maxIterations parameter.
        """
        # Simple transition matrix
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        # Should converge with enough iterations
        result = steadyStateDistribution(P, maxIterations=1000)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        
        # Should raise RuntimeError with too few iterations
        # Note: This test might fail if the matrix converges very quickly
        with pytest.raises(RuntimeError):
            steadyStateDistribution(P, tolerance=1e-15, maxIterations=1)

    def testLargerMatrix(self):
        """
        Test with a larger transition matrix.
        
        This test verifies that the function can handle larger matrices efficiently.
        """
        # Create a larger random transition matrix
        n = 50
        P = np.random.rand(n, n)
        # Make it row-stochastic
        P = P / P.sum(axis=1, keepdims=True)
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testNonSquareMatrix(self):
        """
        Test with a non-square matrix, which should raise a ValueError.
        
        A valid transition matrix must be square.
        """
        # Non-square matrix
        P = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        
        with pytest.raises(ValueError, match="Transition matrix must be square"):
            steadyStateDistribution(P)

    def testNegativeEntries(self):
        """
        Test with a matrix containing negative entries, which should raise a ValueError.
        
        A valid transition matrix must have non-negative entries.
        """
        # Matrix with negative entries
        P = np.array([[0.7, 0.3], [-0.1, 1.1]])
        
        with pytest.raises(ValueError, match="Transition matrix must have non-negative entries"):
            steadyStateDistribution(P)

    def testRowsNotSumToOne(self):
        """
        Test with a matrix whose rows don't sum to 1, which should raise a ValueError.
        
        A valid transition matrix must be row-stochastic (rows sum to 1).
        """
        # Matrix with rows not summing to 1
        P = np.array([[0.7, 0.2], [0.4, 0.5]])
        
        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            steadyStateDistribution(P)

    def testNearlySingularMatrix(self):
        """
        Test with a nearly singular matrix.
        
        This tests the numerical stability of the algorithm with matrices that
        are close to being singular.
        """
        # Create a nearly singular matrix
        P = np.array([
            [0.999999, 0.000001],
            [0.000001, 0.999999]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Expected: nearly uniform distribution
        expected = np.array([0.5, 0.5])
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testKnownAnalyticalSolution(self):
        """
        Test with a matrix that has a known analytical solution.
    
        This test verifies that the function computes the correct steady-state
        distribution for a matrix with a known analytical solution.
        """
        # Matrix with known analytical solution
        P = np.array([
            [0.5, 0.25, 0.25],
            [0.5, 0.0, 0.5],
            [0.25, 0.25, 0.5]
        ])
        
        # Correct analytical solution: π = [0.4, 0.2, 0.4]
        expected = np.array([0.4, 0.2, 0.4])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        assert np.allclose(result, expected, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)

    def testOneDimensionalMatrix(self):
        """
        Test with a 1x1 matrix.
        
        A 1x1 matrix represents a Markov chain with a single state.
        The steady-state distribution should be [1.0].
        """
        # 1x1 matrix
        P = np.array([[1.0]])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Expected: [1.0]
        expected = np.array([1.0])
        
        assert np.array_equal(result, expected)

    def testPeriodicMatrix(self):
        """
        Test with a periodic transition matrix.
        
        A periodic Markov chain has a period greater than 1.
        The power iteration method should still converge to the correct steady-state.
        """
        # Periodic matrix with period 2
        P = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)
        
        # Expected: uniform distribution within each communicating class
        # States 0,1 form one class and states 2,3 form another
        assert np.isclose(result[0], result[1], atol=1e-5)
        assert np.isclose(result[2], result[3], atol=1e-5)

    def testFloatingPointPrecision(self):
        """
        Test the function's handling of floating-point precision issues.
        
        This test creates a matrix with entries that require careful handling
        of floating-point arithmetic to maintain row-stochasticity.
        """
        # Matrix with potential floating-point precision issues
        P = np.array([
            [1/3, 1/3, 1/3],
            [0.1, 0.2, 0.7],
            [0.01, 0.99, 0.0]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testExtremeProbabilityDifferences(self):
        """
        Test with a matrix containing extreme differences in transition probabilities.
        
        This tests the numerical stability of the algorithm with matrices that
        have very large and very small probabilities.
        """
        # Matrix with extreme probability differences
        P = np.array([
            [0.999999, 1e-6, 0],
            [1e-6, 0.999998, 1e-6],
            [0, 1e-6, 0.999999]
        ])
        
        # Compute steady-state distribution
        result = steadyStateDistribution(P)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-5)
        assert np.isclose(np.sum(result), 1.0)

    def testNormalization(self):
        """
        Test that the function correctly normalizes the steady-state distribution.
        
        This test verifies that the function properly normalizes the result to
        ensure it's a valid probability distribution (sums to 1).
        """
        # Create a transition matrix that converges quickly but might need normalization
        P = np.array([
            [0.99, 0.01],
            [0.01, 0.99]
        ])
        
        # Set a very tight tolerance to force more iterations
        result = steadyStateDistribution(P, tolerance=1e-14)
        
        # Check that the result sums exactly to 1
        assert np.isclose(np.sum(result), 1.0, atol=1e-15)
        
        # Verify the steady-state equation: π = π·P
        assert np.allclose(np.dot(result, P), result, rtol=1e-10)
