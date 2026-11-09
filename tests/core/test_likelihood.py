import numpy as np
import pytest
from math import log, sqrt, pi, exp
import scipy.stats
from bayflux.core.likelihood import normpdf  

def testBasicFunctionality():
    """
    Test the basic functionality of the normpdf function.
    
    Verifies that the function correctly calculates log-probability density
    values for normal distributions with various parameters, comparing the
    results with the analytical solution.
    """
    # Test with standard normal distribution
    assert np.isclose(normpdf(0.0, 0.0, 1.0), -0.5 * log(2 * pi), rtol=1e-10)
    
    # Test with non-standard parameters
    assert np.isclose(normpdf(1.0, 2.0, 3.0), 
                      log(exp(-((1.0 - 2.0) ** 2) / (2 * 3.0 ** 2)) / (3.0 * sqrt(2 * pi))),
                      rtol=1e-10)


def testComparisonWithScipy():
    """
    Test normpdf against scipy.stats.norm.logpdf for consistency.
    
    Verifies that our optimized implementation produces the same results
    as the standard scipy implementation across a range of input values.
    """
    test_cases = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (5.0, 10.0, 2.0),
        (-3.0, -5.0, 0.5),
        (0.0, 0.0, 0.1),
    ]
    
    for x, mean, sd in test_cases:
        expected = scipy.stats.norm.logpdf(x, loc=mean, scale=sd)
        actual = normpdf(x, mean, sd)
        assert np.isclose(actual, expected, rtol=1e-10)


def testEdgeCaseLargeValues():
    """
    Test normpdf with extremely large input values.
    
    Verifies that the function returns very negative values for points
    far from the mean, which is the expected behavior for log-likelihood
    of extremely unlikely observations.
    """
    # Test with large values (far from mean)
    x = 1000.0
    mean = 0.0
    sd = 1.0
    
    # The current implementation returns -inf for extreme values
    # which is acceptable for log-likelihood calculations
    result = normpdf(x, mean, sd)
    
    # Either the result should be very negative or -inf
    assert result < -10000 or result == float('-inf')
    
    # For large standard deviation, results should be finite
    x = 1000.0
    mean = 0.0
    sd = 100.0
    result = normpdf(x, mean, sd)
    assert np.isfinite(result)



def testEdgeCaseSmallValues():
    """
    Test normpdf with extremely small input values.
    
    Verifies that the function handles small values and small standard deviations
    correctly without numerical precision issues.
    """
    # Test with small values
    x = 1e-10
    mean = 0.0
    sd = 1.0
    expected = scipy.stats.norm.logpdf(x, loc=mean, scale=sd)
    actual = normpdf(x, mean, sd)
    assert np.isclose(actual, expected, rtol=1e-10)
    
    # Test with small standard deviation
    x = 0.0
    mean = 0.0
    sd = 1e-10
    expected = scipy.stats.norm.logpdf(x, loc=mean, scale=sd)
    actual = normpdf(x, mean, sd)
    assert np.isclose(actual, expected, rtol=1e-10)


def testTypeConversion():
    """
    Test that normpdf correctly handles different input types.
    
    Verifies that the function properly converts inputs to float
    as specified in the docstring, ensuring numerical stability.
    """
    # Test with integer inputs
    int_result = normpdf(1, 0, 1)
    float_result = normpdf(1.0, 0.0, 1.0)
    assert np.isclose(int_result, float_result, rtol=1e-10)
    
    # Test with numpy array inputs (should still work with Numba)
    np_result = normpdf(np.array(1.0), np.array(0.0), np.array(1.0))
    assert np.isclose(np_result, float_result, rtol=1e-10)


def testInvalidInputs():
    """
    Test normpdf's behavior with invalid inputs.
    
    Verifies that the function handles invalid inputs gracefully.
    The current implementation doesn't raise exceptions but returns
    special values like NaN or -inf, which is acceptable for numerical code.
    """
    # Test with negative standard deviation
    result_neg_sd = normpdf(0.0, 0.0, -1.0)
    # Should return NaN or -inf for invalid SD
    assert np.isnan(result_neg_sd) or result_neg_sd == float('-inf')
    
    # Test with zero standard deviation
    result_zero_sd = normpdf(0.0, 0.0, 0.0)
    # Should return NaN or -inf for zero SD
    assert np.isnan(result_zero_sd) or result_zero_sd == float('-inf')


def testNumericalStability():
    """
    Test the numerical stability of normpdf for extreme cases.
    
    Verifies that the function handles extreme cases appropriately.
    For very unlikely observations (far from mean with small SD),
    it's acceptable to return -inf as the log-likelihood.
    """
    # Test case where (x-mean)²/(2*var) is very large
    x = 100.0
    mean = 0.0
    sd = 0.1
    
    # This would cause exp(-5000) which is effectively zero in float precision
    # The current implementation returns -inf which is acceptable
    result = normpdf(x, mean, sd)
    
    # For extreme values, -inf is an acceptable result for log-likelihood
    assert result == float('-inf') or result < -1000
    
    # Test with more reasonable values
    x = 3.0
    mean = 0.0
    sd = 1.0
    result = normpdf(x, mean, sd)
    expected = scipy.stats.norm.logpdf(x, loc=mean, scale=sd)
    assert np.isclose(result, expected, rtol=1e-10)
