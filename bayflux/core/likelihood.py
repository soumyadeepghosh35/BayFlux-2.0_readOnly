"""
Likelihood Calculation Module for BayFlux Bayesian Flux Analysis.

This module provides optimized functions for calculating log-likelihood values in 
Bayesian metabolic flux analysis. It implements fast numerical methods for evaluating 
the probability that experimental mass distribution vector (MDV) measurements are 
consistent with simulated flux distributions from metabolic models.

The module is designed for high-performance MCMC sampling applications where 
likelihood calculations must be performed thousands of times per second. Key 
optimizations include Numba JIT compilation for critical mathematical functions 
and efficient comparison algorithms for experimental vs. simulated data.

Key Features:
- JIT-compiled normal probability density function for speed
- Comprehensive likelihood calculation for flux vector evaluation
- Robust error handling for numerical stability
- Integration with BayFlux EMU network simulation
- Support for substrate labeling experiments and isotope tracing

The likelihood calculation follows standard Bayesian inference principles where 
the likelihood P(data|model) quantifies how well a proposed flux distribution 
explains the observed experimental measurements, accounting for measurement 
uncertainty and noise.

Dependencies:
- NumPy for numerical array operations
- Numba for JIT compilation of mathematical functions
- SciPy for statistical distributions (alternative implementations)
- BayFlux for isotope labeling simulation

Author: BayFlux Development Team
License: See license.txt
Version: 2.0
"""

import numpy as np
import math
from numba import njit, types
from numba.extending import overload
import bayflux
from typing import Dict, Any, Union, Tuple
from scipy.stats import norm  # DEVELOPER: Alternative implementation preserved for comparison

# Global constants for numerical stability and mathematical calculations
TWO_PI = 2.0 * np.pi  # Mathematical constant for normal distribution calculations
NUMERICAL_STABILITY_EPSILON = 1e-7  # Small value added to prevent division by zero in flux calculations
MIN_LOG_LIKELIHOOD = np.finfo(np.float32).min  # Minimum representable log-likelihood for error cases
SQRT_TWO_PI = math.sqrt(TWO_PI)  # Precomputed square root for performance optimization
LOG_SQRT_TWO_PI = math.log(SQRT_TWO_PI)  # Precomputed log normalization term of the log-pdf


def asFloatScalar(value):
    """
    Return value as a Python float.

    Accepts Python numbers, numpy scalars and zero-dimensional numpy arrays.
    The last case is common when a single measurement is pulled out of a
    numpy or pandas object, and Numba cannot apply float() to it directly,
    so the overload below gives it an explicit implementation. The scalar
    path compiles down to the same instruction as a bare float() call, so
    there is no cost in the MCMC inner loop.
    """
    return float(np.asarray(value).item())


@overload(asFloatScalar)
def asFloatScalarImpl(value):
    """Numba implementation selector for asFloatScalar."""
    if isinstance(value, types.Array):
        if value.ndim != 0:
            raise TypeError(
                "asFloatScalar expects a scalar or a zero-dimensional array"
            )
        return lambda value: float(value[()])
    return lambda value: float(value)


@njit
def normpdf(x: float, mean: float, sd: float) -> float:
    """
    Calculate the natural logarithm of the normal probability density function.

    This function computes the log of the normal (Gaussian) probability density
    at point x for a distribution with given mean and standard deviation.
    It is optimized for speed using Numba's just-in-time compilation and is
    significantly faster than scipy.stats.norm.logpdf() in MCMC loops.

    In the context of BayFlux, this function is primarily used to calculate 
    log-likelihood values when comparing experimental mass distribution vectors 
    (MDVs) with simulated results from metabolic flux models. Each experimental 
    measurement contributes to the overall likelihood through this function.

    Parameters
    ----------
    x : float
        The observed value, typically an experimental MDV measurement from 
        LC-MS analysis. Represents the fraction of labeled isotopologue.
    mean : float
        The expected value, typically a simulated MDV value computed from 
        the current flux vector proposal using EMU network simulation.
    sd : float
        The standard deviation representing measurement error or experimental
        uncertainty. Must be strictly positive. Typically estimated from
        replicate measurements or instrument specifications. A value of zero
        or less does not define a distribution and yields NaN.

    Returns
    -------
    float
        The natural logarithm of the normal probability density function,
        which represents the log-likelihood contribution of this measurement 
        to the overall model fit. More negative values indicate worse fit.
        Returns NaN if sd is zero or negative.

    Notes
    -----
    The function implements the formula:
    log(f(x)) = -log(σ·√(2π)) - (x-μ)²/(2σ²)
    
    This is mathematically equivalent to scipy.stats.norm.logpdf() but optimized 
    for repeated evaluation in MCMC sampling scenarios. The implementation:
    
    1. Coerces the inputs to float, accepting Python numbers, numpy scalars
       and zero-dimensional numpy arrays
    2. Rejects a non-positive standard deviation by returning NaN
    3. Forms the standardised residual z = (x - mean) / sd
    4. Returns -log(sqrt(2*pi)) - log(sd) - z^2/2

    The evaluation stays in the log domain throughout. An earlier version
    computed exp(-z^2/2) first and took the logarithm of the result, which
    underflows to zero once the residual reaches 38.58 standard deviations
    and returns -inf from there on. A single -inf log-likelihood is enough to
    stall an MCMC chain, so the log domain form is used instead. It is also
    what scipy.stats.norm.logpdf computes, to full double precision.


    See Also
    --------
    scipy.stats.norm.logpdf : Standard library equivalent (slower)
    logLikelihood : Main likelihood calculation function using this component
    """
    # Coerce the inputs so that ints, numpy scalars and zero-dimensional
    # numpy arrays are all accepted
    xValue = asFloatScalar(x)
    meanValue = asFloatScalar(mean)
    sdValue = asFloatScalar(sd)

    # A standard deviation of zero or less does not define a distribution.
    # NaN propagates through the summed log-likelihood and is visible to the
    # caller, whereas an arbitrary finite value would be silently wrong.
    if sdValue <= 0.0:
        return np.nan

    # Standardised residual, then the log-pdf evaluated in the log domain
    zScore = (xValue - meanValue) / sdValue
    return -LOG_SQRT_TWO_PI - math.log(sdValue) - 0.5 * zScore * zScore


def logLikelihood(
    fluxVector: np.ndarray, 
    lcmsResults: Any, 
    compiledData: Dict[str, Any], 
    substrateLabelingDict: Dict[str, Any], 
    errorSD: float
) -> float:
    """Log likelihood of a flux vector given measured mass distribution vectors.

    The labeling for the proposed fluxes is simulated with the EMU network and
    compared with each measured MDV entry under a normal error model.

    Parameters
    ----------
    fluxVector : np.ndarray
        Fluxes to evaluate, one per variable. A small constant is added so that
        no entry is exactly zero.
    lcmsResults : MassDistribution
        Measured MDVs. Its data attribute maps each EMU to a list of MDVs.
    compiledData : list
        EMU network compiled by bayflux.emuCompile.
    substrateLabelingDict : dict
        Labeling of the input substrates.
    errorSD : float
        Standard deviation of the measurement error. Must be positive.

    Returns
    -------
    float
        Summed log likelihood. A failed linear solve returns the lowest float
        value so the sampler rejects the proposal instead of stopping.

    Raises
    ------
    AssertionError
        If a measured EMU is missing from the compiled network.
    """
    # Add small epsilon to prevent exact zeros in flux vector
    # Exact zeros can cause numerical instability in downstream matrix operations
    # and violate the assumption of feasible flux space exploration
    adjustedFluxVector = fluxVector + NUMERICAL_STABILITY_EPSILON

    # Simulate isotope labeling patterns using EMU network methodology
    try:
        # Call BayFlux simulation engine to compute expected MDV values
        # This performs the forward simulation: flux → isotope distribution
        simulationResults = bayflux.simulateLabeling(
            compiledData, adjustedFluxVector, substrateLabelingDict
        )
    except np.linalg.LinAlgError:
        # Handle linear algebra failures gracefully during simulation
        # This can occur with poorly conditioned flux distributions or 
        # numerical issues in the EMU network matrix operations
        print('LinAlgError encountered during isotope simulation')
        return MIN_LOG_LIKELIHOOD  # Return extremely low likelihood to reject this proposal

    # Initialize cumulative log-likelihood accumulator
    cumulativeLogLikelihood = 0.0

    # Compare experimental measurements with simulation results for each EMU
    for emu, experimentalMdvs in lcmsResults.data.items():
        # Generate hash for efficient EMU lookup in compiled data structures
        emuHash = hash(emu)
        emuFound = False
        
        # Search through all EMU size classes for matching experimental EMU
        for singleSizeData in compiledData:
            # Check if current EMU exists in this size class
            if emuHash in singleSizeData['matrixCoords']['internalEMUs']['hashes']:
                # Find the index position of this EMU in the size class
                emuIndex = singleSizeData['matrixCoords']['internalEMUs']['hashes'].index(emuHash)
                emuFound = True
                
                # Extract simulated results for this EMU from the appropriate size class
                simulatedMdvForEmu = simulationResults[singleSizeData['size']][emuIndex]
                
                # Process all experimental replicates for this EMU
                for experimentalMdv in experimentalMdvs:
                    # Compare experimental vs simulated values point-by-point
                    # Each experimental_value, simulated_value pair contributes to likelihood
                    for experimentalValue, simulatedValue in zip(experimentalMdv, simulatedMdvForEmu):
                        # Calculate log-likelihood contribution using normal distribution model
                        # This assumes measurement errors are normally distributed with known variance
                        likelihoodContribution = normpdf(experimentalValue, simulatedValue, errorSD)
                        cumulativeLogLikelihood += likelihoodContribution
                        
                        # DEVELOPER: Alternative implementation using SciPy for comparison
                        # This commented line shows the equivalent calculation using standard library
                        # Performance testing shows normpdf() is significantly faster in MCMC loops
                        # likelihoodContribution = norm.logpdf(experimentalValue, loc=simulatedValue, scale=errorSD)
                
                # Break inner loop once EMU is found and processed
                break
        
        # Verify that every experimental EMU was found in the compiled network
        # This assertion ensures data completeness and catches configuration errors
        assert emuFound == True, f'Missing experimental EMU in simulated results: {repr(emu)}'
    
    return cumulativeLogLikelihood



