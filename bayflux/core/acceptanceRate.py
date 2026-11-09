"""
Acceptance Rate Analysis Module for MP-MCMC Sampling Diagnostics.

This module provides utilities for analyzing the performance and convergence 
characteristics of Multi-Proposal Markov Chain Monte Carlo (MP-MCMC) sampling 
algorithms. The primary focus is on computing acceptance rates, which serve as 
critical diagnostic metrics for assessing the efficiency and mixing properties 
of MCMC samplers.

Acceptance rate analysis is essential for:
- Evaluating sampler performance and tuning parameters
- Detecting poor mixing or convergence issues
- Optimizing proposal distributions and step sizes
- Ensuring reliable posterior inference in Bayesian analysis

The module implements robust statistical methods with comprehensive error handling
and validation to support production-level MCMC diagnostics in scientific computing
applications, particularly within the BayFlux metabolic flux analysis framework.

Key Features:

- Burn-in period handling for unbiased rate estimation
- Comprehensive input validation and error checking
- Support for various array-like input formats
- Production-ready error handling and logging
- Statistical guidelines for optimal acceptance rates

Mathematical Background:

The acceptance rate α is computed as:
α = (1/N) ∑(i=burn+1 to N) I(accepted_i)
where I(·) is the indicator function and N is the total number of samples.

References:

- Schwedes et al. (2021): "Multi-Proposal MCMC Methods"
  https://proceedings.mlr.press/v130/schwedes21a.html
- Roberts & Rosenthal (2001): "Optimal scaling for various Metropolis-Hastings algorithms"

Dependencies:

- NumPy for efficient numerical array operations
- Math for mathematical functions and constants

Author: BayFlux Development Team
License: See license.txt
Version: 2.0
"""

import numpy as np
import math
from typing import Union, List, Tuple, Any
import warnings

# Global constants for acceptance rate analysis and MCMC diagnostics
DEFAULT_BURN_IN = 25  # Standard burn-in period for MCMC convergence
MIN_ACCEPTANCE_RATE = 0.2  # Lower bound for healthy MCMC acceptance rates
MAX_ACCEPTANCE_RATE = 0.5  # Upper bound for healthy MCMC acceptance rates
OPTIMAL_ACCEPTANCE_RATE = 0.234  # Theoretical optimal rate for high-dimensional problems
MINIMUM_SAMPLES_REQUIRED = 10  # Minimum samples needed for meaningful rate calculation


def computeAcceptanceRate(
    arrayOfAcceptedSamples: Union[List[Union[int, float]], np.ndarray, List[List[Union[int, float]]]],
    burnIn: Union[int, float] = DEFAULT_BURN_IN
) -> float:
    """Mean acceptance rate of an MP-MCMC run, after discarding burn-in.

    Parameters
    ----------
    arrayOfAcceptedSamples : list or np.ndarray
        Acceptance indicators (1 accepted, 0 rejected), either flat or as a list
        of arrays that are concatenated. Non-binary values are averaged as given.
    burnIn : int or float, optional
        Number of leading samples to discard, default 25. A float is rounded up.

    Returns
    -------
    float
        Mean acceptance rate between 0 and 1.

    Raises
    ------
    ValueError
        If the input is empty or cannot be converted to numbers, or if burnIn is
        negative or not smaller than the number of samples.
    TypeError
        If burnIn is not convertible to an integer, or the input is not
        array-like.

    Warns
    -----
    UserWarning
        If fewer than 10 samples remain after burn-in.
    """
    # Input validation: Check for empty inputs across different data types
    if isinstance(arrayOfAcceptedSamples, (list, tuple)):
        if len(arrayOfAcceptedSamples) == 0:
            raise ValueError("Input array cannot be empty")
        
        # Handle nested lists/arrays (e.g., multiple chains)
        if all(isinstance(item, (list, tuple, np.ndarray)) for item in arrayOfAcceptedSamples):
            # Check for zero-dimensional and empty nested arrays. len() raises
            # TypeError on a zero-dimensional array, so test ndim first.
            for item in arrayOfAcceptedSamples:
                if isinstance(item, np.ndarray) and item.ndim == 0:
                    raise ValueError("Input arrays cannot be empty or zero-dimensional")
                if len(item) == 0:
                    raise ValueError("Input contains empty sub-arrays")
    elif isinstance(arrayOfAcceptedSamples, np.ndarray):
        if arrayOfAcceptedSamples.size == 0:
            raise ValueError("Input array cannot be empty")
    else:
        # Attempt to check length for other array-like objects
        try:
            if len(arrayOfAcceptedSamples) == 0:
                raise ValueError("Input array cannot be empty")
        except (TypeError, AttributeError):
            raise TypeError("Input must be a non-empty array-like object")
    
    # Burn-in parameter validation and type conversion
    try:
        if isinstance(burnIn, float):
            # Round up floating-point burn-in values to ensure integer sample count
            burnIn = math.ceil(burnIn)
        else:
            burnIn = int(burnIn)
    except (TypeError, ValueError) as e:
        raise TypeError(f"burnIn must be an integer or convertible to integer: {e}")
    
    # Validate burn-in value constraints
    if burnIn < 0:
        raise ValueError("burnIn cannot be negative")
    
    # Convert input to flattened numeric array for uniform processing
    try:
        # Handle different input formats by attempting concatenation
        if isinstance(arrayOfAcceptedSamples, (list, tuple)) and \
           any(isinstance(item, (list, tuple, np.ndarray)) for item in arrayOfAcceptedSamples):
            # Multi-dimensional input: concatenate all sub-arrays
            concatenatedSamples = np.concatenate([
                np.asarray(arr, dtype=float) for arr in arrayOfAcceptedSamples
            ])
        else:
            # Single-dimensional input: direct conversion
            concatenatedSamples = np.asarray(arrayOfAcceptedSamples, dtype=float)
    except ValueError as e:
        if "zero-dimensional" in str(e).lower():
            raise ValueError("Input arrays cannot be empty or zero-dimensional")
        elif "could not broadcast" in str(e).lower():
            raise ValueError("Input contains arrays with incompatible shapes")
        else:
            raise ValueError(f"Failed to convert input to numeric array: {e}")
    except Exception as e:
        raise TypeError(f"Input data cannot be converted to numeric format: {e}")
    
    # Validate sample count vs burn-in period
    totalSamples = len(concatenatedSamples)
    if burnIn >= totalSamples:
        raise ValueError(
            f"burnIn ({burnIn}) must be less than total samples ({totalSamples}). "
            f"Cannot discard all available data."
        )
    
    # Check for minimum sample requirement after burn-in
    samplesAfterBurnIn = totalSamples - burnIn
    if samplesAfterBurnIn < MINIMUM_SAMPLES_REQUIRED:
        warnings.warn(
            f"Very few samples ({samplesAfterBurnIn}) remain after burn-in. "
            f"Consider reducing burn-in period or collecting more samples for "
            f"reliable acceptance rate estimation.",
            UserWarning
        )
    
    # Extract post-burn-in samples for acceptance rate calculation
    acceptedValues = concatenatedSamples[burnIn:]
    
    
    # Calculate mean acceptance rate
    acceptanceRate = np.mean(acceptedValues)
    
    return acceptanceRate


