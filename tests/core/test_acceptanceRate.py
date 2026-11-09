import numpy as np
import pytest
from bayflux.core.acceptanceRate import computeAcceptanceRate  


def testBasicFunctionality():
    """
    Test the basic functionality of the computeAcceptanceRate function.
    
    Verifies that the function correctly calculates the acceptance rate
    for a simple array of accepted/rejected samples, properly accounting
    for the burn-in period.
    """
    # Simple case with 50% acceptance rate
    samples = [np.array([0, 1, 0, 1, 0, 1])]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 0.5)
    
    # With burn-in period
    samples = [np.array([0, 0, 0, 1, 1, 1])]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=3), 1.0)


def testMultipleChains():
    """
    Test the function with multiple chains of samples.
    
    Verifies that the function correctly concatenates and processes
    multiple chains of accepted/rejected samples.
    """
    # Multiple chains with different acceptance patterns
    samples = [
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0])
    ]
    # After concatenation: [0, 0, 1, 1, 1, 1, 0, 0], with burnIn=2
    # Remaining values: [1, 1, 1, 1, 0, 0], acceptance rate = 4/6 = 0.6667
    assert np.isclose(computeAcceptanceRate(samples, burnIn=2), 4/6)


def testEmptyInput():
    """
    Test the function's behavior with empty input arrays.
    
    Verifies that the function raises appropriate exceptions when given
    empty arrays, which could occur if no samples were collected.
    """
    # Empty list of chains
    with pytest.raises(ValueError):
        computeAcceptanceRate([], burnIn=0)
    
    # List with empty array
    with pytest.raises(ValueError):
        computeAcceptanceRate([np.array([])], burnIn=0)


def testAllAccepted():
    """
    Test the function when all samples are accepted.
    
    Verifies that the function correctly calculates a 100% acceptance rate
    when all proposals are accepted.
    """
    samples = [np.array([1, 1, 1, 1, 1, 1])]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 1.0)
    
    # With burn-in
    assert np.isclose(computeAcceptanceRate(samples, burnIn=2), 1.0)


def testAllRejected():
    """
    Test the function when all samples are rejected.
    
    Verifies that the function correctly calculates a 0% acceptance rate
    when all proposals are rejected.
    """
    samples = [np.array([0, 0, 0, 0, 0, 0])]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 0.0)
    
    # With burn-in
    assert np.isclose(computeAcceptanceRate(samples, burnIn=2), 0.0)


def testBurnInLargerThanSamples():
    """
    Test the function when burn-in exceeds the number of samples.
    
    Verifies that the function raises an appropriate exception when the burn-in period
    is larger than or equal to the total number of samples available.
    """
    samples = [np.array([0, 1, 0, 1, 0])]
    
    # When burn-in equals sample size
    with pytest.raises(ValueError):
        computeAcceptanceRate(samples, burnIn=5)
    
    # When burn-in exceeds sample size
    with pytest.raises(ValueError):
        computeAcceptanceRate(samples, burnIn=10)


def testNonBinaryValues():
    """
    Test the function with non-binary values.
    
    Verifies that the function still works with values that are not
    strictly 0 or 1 but can be interpreted as numeric values.
    """
    # Using True/False instead of 1/0
    samples = [np.array([True, False, True, True, False])]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 0.6)
    
    # Using float values
    samples = [np.array([0.0, 1.0, 2.0, 0.5, 0.0])]
    expected_rate = 3.5/5  # Sum of values divided by count
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), expected_rate)


def testDifferentInputTypes():
    """
    Test the function with different types of input arrays.
    
    Verifies that the function works correctly with different array-like
    inputs, such as Python lists, NumPy arrays, and nested structures.
    """
    # Python list instead of numpy array
    samples = [[0, 1, 0, 1, 0]]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 0.4)
    
    # Mixed types
    samples = [
        np.array([0, 1, 0]),
        [1, 0, 1]
    ]
    assert np.isclose(computeAcceptanceRate(samples, burnIn=0), 0.5)


def testNegativeBurnIn():
    """
    Test the function with negative burn-in values.
    
    Verifies that the function raises an appropriate exception when given
    negative burn-in values, which are invalid.
    """
    samples = [np.array([0, 1, 0, 1, 0])]
    
    # Negative burn-in should raise an error
    with pytest.raises(ValueError):
        computeAcceptanceRate(samples, burnIn=-1)


def testLargeArrays():
    """
    Test the function with large input arrays.
    
    Verifies that the function performs correctly and efficiently
    with large arrays, which is important for real-world MCMC applications.
    """
    # Generate large random array with known acceptance rate
    np.random.seed(42)  # For reproducibility
    large_array = np.random.choice([0, 1], size=10000, p=[0.7, 0.3])
    samples = [large_array]
    
    # Expected acceptance rate should be close to 0.3
    assert np.isclose(computeAcceptanceRate(samples, burnIn=1000), 0.3, atol=0.02)


def testFloatingPointBurnIn():
    """
    Test the function with floating-point burn-in values.
    
    Verifies that the function correctly handles floating-point burn-in values
    by converting them to integers.
    """
    samples = [np.array([0, 0, 0, 1, 1, 1])]
    
    # Floating-point burn-in should be converted to int
    assert np.isclose(computeAcceptanceRate(samples, burnIn=2.7), 1.0)
    assert np.isclose(computeAcceptanceRate(samples, burnIn=2.2), 1.0)


def testInvalidBurnInType():
    """
    Test the function with invalid burn-in types.
    
    Verifies that the function raises an appropriate exception when given
    burn-in values that cannot be converted to integers.
    """
    samples = [np.array([0, 1, 0, 1, 0])]
    
    # String that can't be converted to int
    with pytest.raises(TypeError):
        computeAcceptanceRate(samples, burnIn="not_a_number")
    
    # Complex number
    with pytest.raises(TypeError):
        computeAcceptanceRate(samples, burnIn=1+2j)
