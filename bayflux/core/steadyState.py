"""
Steady State Distribution Calculation Module for Markov Chain Analysis.

This module provides optimized algorithms for computing steady-state (stationary) 
distributions of finite Markov chains represented by transition probability matrices. 
The implementation is specifically designed for high-performance applications in 
Bayesian MCMC sampling, particularly within the BayFlux metabolic flux analysis framework.

The module implements multiple approaches to steady-state calculation:
1. Power iteration method (primary, JIT-compiled for speed)
2. Eigenvalue decomposition method (alternative, commented for reference)

Key Features:
- Numba JIT compilation for maximum performance in MCMC loops
- Robust numerical validation and error handling
- Damping mechanisms to prevent oscillations during convergence
- Fallback strategies for numerical edge cases
- Comprehensive input validation for production safety

Mathematical Background:
The steady-state distribution π of a transition matrix P satisfies:
π = πP (or equivalently π^T = P^T π^T)

This represents the long-run probability distribution of states in the Markov chain,
independent of the initial state distribution.

Dependencies:
- NumPy for numerical array operations and mathematical functions
- Numba for JIT compilation and performance optimization
- SciPy for eigenvalue computations (in alternative implementation)

Author: BayFlux Development Team
License: [License Information]
Version: 2.0
"""

import numpy as np
from numba import jit
from typing import Tuple, Union

# Global constants for numerical methods and convergence criteria
DEFAULT_TOLERANCE = 1e-8  # Default convergence tolerance for iterative methods
DEFAULT_MAX_ITERATIONS = 1000  # Maximum number of power iteration steps
DAMPING_FACTOR = 0.99  # Damping coefficient to reduce oscillations during convergence
MATRIX_VALIDATION_TOLERANCE = 1e-8  # Tolerance for row sum validation in transition matrices
NORMALIZATION_TOLERANCE = 1e-10  # Tolerance for final probability distribution validation


@jit(nopython=True, cache=True, fastmath=True)
def steadyStateDistribution(
    transitionMatrix: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    maxIterations: int = DEFAULT_MAX_ITERATIONS
) -> np.ndarray:
    """Steady-state distribution of a Markov chain by power iteration.

    Iterates pi(k+1) = pi(k) P from a uniform start, with damping of 0.99 to
    suppress oscillation, until the L2 change between iterations is below
    tolerance. Compiled with Numba because it runs inside the sampling loop.

    Parameters
    ----------
    transitionMatrix : np.ndarray
        Square (n, n) row-stochastic matrix. P[i, j] is the probability of moving
        from state i to state j.
    tolerance : float, optional
        Convergence threshold on the L2 change, default 1e-8.
    maxIterations : int, optional
        Iteration limit, default 1000.

    Returns
    -------
    np.ndarray
        Probability vector of shape (n,) with pi = pi P.

    Raises
    ------
    ValueError
        If the matrix is not square, has negative entries, or has rows that do
        not sum to 1.
    RuntimeError
        If the iteration does not converge within maxIterations.
    """
    # Extract matrix dimensions for validation and algorithm setup
    matrixSize = transitionMatrix.shape[0]

    # Comprehensive input validation to ensure mathematical requirements
    if transitionMatrix.shape[0] != transitionMatrix.shape[1]:
        raise ValueError("Transition matrix must be square")
    
    # Validate stochastic matrix properties: non-negative entries and row sums = 1
    for i in range(matrixSize):
        rowSum = 0.0
        for j in range(matrixSize):
            # Check non-negativity constraint for probability matrix
            if transitionMatrix[i, j] < 0:
                raise ValueError("Transition matrix must have non-negative entries")
            rowSum += transitionMatrix[i, j]
        
        # Verify stochastic property: each row must sum to 1.0
        if abs(rowSum - 1.0) > MATRIX_VALIDATION_TOLERANCE:
            raise ValueError("Transition matrix rows must sum to 1")

    # Initialize probability distribution with uniform starting point
    # This choice is arbitrary since steady state is independent of initial distribution
    currentDistribution = np.ones(matrixSize, dtype=np.float64) / matrixSize
    nextDistribution = np.zeros(matrixSize, dtype=np.float64)

    # Power iteration main loop with damping for numerical stability
    for iterationCount in range(maxIterations):
        # Compute next distribution: π^(k+1) = π^(k) P
        # Manual matrix-vector multiplication for optimal JIT performance
        for i in range(matrixSize):
            nextDistribution[i] = 0.0
            for j in range(matrixSize):
                # Accumulate weighted probabilities from all incoming states
                nextDistribution[i] += transitionMatrix[j, i] * currentDistribution[j]

        # Apply damping to prevent oscillations and improve convergence stability
        # π_damped = α·π_new + (1-α)·π_old where α = DAMPING_FACTOR
        for i in range(matrixSize):
            nextDistribution[i] = (DAMPING_FACTOR * nextDistribution[i] + 
                                 (1 - DAMPING_FACTOR) * currentDistribution[i])

        # Compute L2 norm of difference vector for convergence assessment
        # ||π^(k+1) - π^(k)||₂ measures how much the distribution changed
        differenceNorm = 0.0
        for i in range(matrixSize):
            differenceNorm += (nextDistribution[i] - currentDistribution[i]) ** 2
        differenceNorm = np.sqrt(differenceNorm)
        
        # Check convergence criterion: stop if change is below tolerance
        if differenceNorm < tolerance:
            # Final normalization to ensure exact probability distribution properties
            # This corrects for accumulated floating-point rounding errors
            distributionSum = 0.0
            for i in range(matrixSize):
                distributionSum += nextDistribution[i]
            
            # Normalize to sum exactly to 1.0
            for i in range(matrixSize):
                nextDistribution[i] /= distributionSum
            
            return nextDistribution

        # Update current distribution for next iteration
        # Copy values element-wise for optimal JIT performance
        for i in range(matrixSize):
            currentDistribution[i] = nextDistribution[i]

    # If we reach here, the algorithm failed to converge within maxIterations
    raise RuntimeError("Power iteration did not converge")





