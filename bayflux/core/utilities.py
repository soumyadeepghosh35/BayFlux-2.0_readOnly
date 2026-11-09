"""
System Resource Monitoring and Numerical Precision Analysis Module.

The module addresses two critical aspects of production scientific computing:

1. Resource Monitoring: Real-time tracking of memory and CPU usage to prevent
   system overload and optimize computational resource allocation during intensive
   MCMC sampling operations.

2. Numerical Precision Analysis: Detection and assessment of potential numerical
   instabilities in likelihood calculations, particularly when dealing with extreme
   logLikelihood values that can cause overflow/underflow in exponential computations.

Author: BayFlux Development Team
License: See license.txt
Version: 2.0
"""

import numpy as np
import psutil
import os
import logging
from typing import Dict, Union, Any, Tuple, List

# Global constants for numerical precision analysis and system monitoring
OVERFLOW_THRESHOLD = 700.0  # Log-likelihood difference threshold for exp() overflow
UNDERFLOW_THRESHOLD = -700.0  # Log-likelihood difference threshold for exp() underflow
BYTES_TO_MB_DIVISOR = 1024 ** 2  # Conversion factor from bytes to megabytes
HIGH_RISK_LIKELIHOOD_RANGE = 1400.0  # Range indicating high numerical risk (2 * 700)
MODERATE_RISK_LIKELIHOOD_RANGE = 700.0  # Range indicating moderate numerical risk
LOW_RISK_PERCENTAGE_THRESHOLD = 0.01  # Threshold for classifying numerical risk as low
PRECISION_DECIMAL_PLACES = 3  # Number of decimal places for precision reporting


def logSystemResources() -> None:
    """
    Log current memory and CPU usage of the running process for performance monitoring.
    
    This function captures real-time system resource consumption metrics and logs them
    using the Python logging framework. It provides essential monitoring capabilities
    for resource-intensive MCMC sampling operations, helping to detect memory leaks,
    CPU bottlenecks, and optimize computational resource allocation.
    
    The function reports:

    - Memory Usage: Resident Set Size (RSS) in megabytes - the physical memory
      currently used by the process, excluding swapped memory
    - CPU Usage: System-wide CPU utilization percentage at the time of measurement
    
    This monitoring is particularly valuable during long-running Bayesian sampling
    operations where resource consumption can grow over time or vary significantly
    between different phases of the computation.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Function performs logging operations but does not return values.
        Resource metrics are written to the configured logging destination.

    Raises
    ------
    psutil.NoSuchProcess
        If the current process PID becomes invalid during execution.
    psutil.AccessDenied
        If insufficient permissions to access process information.
    OSError
        If system resource information cannot be retrieved.

    
    See Also
    --------
    psutil.Process : Detailed process monitoring capabilities
    psutil.virtual_memory : System-wide memory statistics
    logging : Python logging framework for production applications
    """
    try:
        # Get current process information using process ID
        currentProcess = psutil.Process(os.getpid())
        
        # Retrieve memory information structure containing RSS and other metrics
        memoryInfo = currentProcess.memory_info()
        
        # Calculate memory usage in MB from bytes (RSS = Resident Set Size)
        memoryUsageMB = memoryInfo.rss / BYTES_TO_MB_DIVISOR
        
        # Get instantaneous system-wide CPU utilization percentage
        cpuUsagePercent = psutil.cpu_percent()
        
        # Log resource metrics using configured logging framework
        logging.info(f"Memory usage: {memoryUsageMB:.2f} MB")
        logging.info(f"CPU usage: {cpuUsagePercent}%")
        
    except psutil.NoSuchProcess:
        logging.error("Process no longer exists - cannot retrieve resource information")
        raise
    except psutil.AccessDenied:
        logging.error("Access denied - insufficient permissions to retrieve process information")
        raise
    except Exception as e:
        logging.error(f"Unexpected error retrieving system resources: {e}")
        raise


def analyzeNumericalPrecision(
    samplesTemp: np.ndarray,
    likelihoods: Union[List[float], np.ndarray],
    numProposals: int
) -> Dict[str, Union[np.ndarray, int, str]]:
    """
    Analyze numerical precision issues in MCMC transition probability calculations.
    
    This function performs comprehensive analysis of potential numerical instabilities
    that can arise when computing transition probabilities in Bayesian MCMC sampling.
    It focuses on detecting cases where log-likelihood differences are so extreme
    that exponential calculations will overflow or underflow, leading to invalid
    probabilities and sampling failures.
    
    The analysis examines all pairwise log-likelihood differences between current
    and proposed states, categorizes them by numerical risk level, and provides
    actionable recommendations for maintaining computational stability.

    Parameters
    ----------
    samplesTemp : np.ndarray
        Array of sample flux vectors with shape (numProposals + 1, n_reactions).
        Row 0 contains the current state, subsequent rows contain proposal states
        from parallel MCMC workers. Used for context in the analysis.
    likelihoods : Union[List[float], np.ndarray]
        Log-likelihood values corresponding to each flux vector in samplesTemp.
        Shape should be (numProposals + 1,). These are the values that will be
        used in exponential calculations for acceptance probability computations.
    numProposals : int
        Number of proposal states (excluding current state). Used for validation
        and reporting purposes in the precision analysis.

    Returns
    -------
    Dict[str, Union[np.ndarray, int, str]]
        Comprehensive analysis results containing:
        
        - 'log_diffs' : np.ndarray
            All pairwise log-likelihood differences (j-i for all i≠j)
        - 'overflow_cases' : int
            Number of differences exceeding OVERFLOW_THRESHOLD (700)
        - 'underflow_cases' : int  
            Number of differences below UNDERFLOW_THRESHOLD (-700)
        - 'safe_cases' : int
            Number of differences in computationally safe range
        - 'risk_level' : str
            Overall risk assessment: 'LOW', 'MODERATE', or 'HIGH'

    
    See Also
    --------
    quickPrecisionCheck : Rapid risk assessment for likelihood ranges
    numpy.exp : Exponential function subject to overflow/underflow
    """
    # Ensure numerical stability by converting to high-precision float64
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    
    # DEVELOPER: Comprehensive diagnostic output section
    # The following commented print statements provide detailed analysis output
    # that is invaluable during development, debugging, and algorithm validation.
    # They are preserved in commented form to maintain production performance
    # while keeping diagnostic capabilities readily available when needed.
    
    # Header and basic problem setup information
    #print("="*60)
    #print("NUMERICAL PRECISION ANALYSIS")
    #print("="*60)
    
    # Basic statistical summary of the likelihood landscape
    # These metrics help understand the scale and distribution of likelihood values
    #print(f"Number of proposals: {numProposals}")
    #print(f"Number of samples: {len(likelihoods)}")
    #print(f"Log-likelihood range: [{np.min(likelihoods):.3f}, {np.max(likelihoods):.3f}]")
    #print(f"Log-likelihood span: {np.max(likelihoods) - np.min(likelihoods):.3f}")
    #print(f"Log-likelihood std: {np.std(likelihoods):.3f}")
    
    # Compute all pairwise log-likelihood differences for transition probability analysis
    # These differences directly determine the arguments to exp() in acceptance calculations
    logDifferences = []
    for i in range(len(likelihoods)):
        for j in range(len(likelihoods)):
            if i != j:
                # Calculate log(P_j) - log(P_i) which appears in acceptance ratio
                logDifferences.append(likelihoods[j] - likelihoods[i])
    
    logDifferences = np.array(logDifferences)
    
    # Statistical analysis of the pairwise differences
    #print(f"\nPairwise log-likelihood differences:")
    #print(f"  Range: [{np.min(logDifferences):.3f}, {np.max(logDifferences):.3f}]")
    #print(f"  Mean: {np.mean(logDifferences):.3f}")
    #print(f"  Std: {np.std(logDifferences):.3f}")
    
    # Categorize differences by numerical risk level based on exponential limits
    overflowCases = np.sum(logDifferences > OVERFLOW_THRESHOLD)
    underflowCases = np.sum(logDifferences < UNDERFLOW_THRESHOLD)
    safeCases = np.sum((logDifferences >= UNDERFLOW_THRESHOLD) & 
                      (logDifferences <= OVERFLOW_THRESHOLD))
    
    totalCases = len(logDifferences)
    
    # Detailed breakdown of numerical precision risks
    #print(f"\nNumerical precision assessment:")
    #print(f"  Safe cases (exp computable): {safeCases}/{totalCases} ({safeCases/totalCases*100:.1f}%)")
    #print(f"  Overflow cases (exp → ∞): {overflowCases}/{totalCases} ({overflowCases/totalCases*100:.1f}%)")
    #print(f"  Underflow cases (exp → 0): {underflowCases}/{totalCases} ({underflowCases/totalCases*100:.1f}%)")
    
    # Determine overall risk level and provide actionable recommendations
    problematicCases = overflowCases + underflowCases
    riskPercentage = problematicCases / totalCases
    
    if overflowCases == 0 and underflowCases == 0:
        riskLevel = "LOW"
        recommendation = "No numerical issues detected"
    elif riskPercentage < LOW_RISK_PERCENTAGE_THRESHOLD:
        riskLevel = "MODERATE"
        recommendation = "Minor numerical issues, consider monitoring"
    else:
        riskLevel = "HIGH"
        recommendation = "Significant numerical issues, consider log-space calculations"
    
    # Final assessment and recommendations
    #print(f"\nRisk Assessment: {riskLevel}")
    #print(f"Recommendation: {recommendation}")
    
    # Return comprehensive analysis results for programmatic use
    return {
        'log_diffs': logDifferences,
        'overflow_cases': overflowCases,
        'underflow_cases': underflowCases,
        'safe_cases': safeCases,
        'risk_level': riskLevel
    }


def quickPrecisionCheck(likelihoods: Union[List[float], np.ndarray]) -> str:
    """
    Perform rapid assessment of potential numerical precision issues.
    
    This function provides a fast, lightweight check for numerical instability
    risks based on the range of log-likelihood values. It serves as a quick
    screening tool that can be called frequently during MCMC sampling without
    significant computational overhead.
    
    The assessment is based on the principle that large ranges in log-likelihood
    values increase the probability of extreme pairwise differences that lead
    to overflow or underflow in exponential calculations.

    Parameters
    ----------
    likelihoods : Union[List[float], np.ndarray]
        Collection of log-likelihood values to assess for numerical risk.
        Can be any array-like structure containing numeric values.

    Returns
    -------
    str
        Risk level classification:
        - 'LOW': Range ≤ 700, minimal overflow/underflow risk
        - 'MODERATE': Range 700-1400, some risk of numerical issues  
        - 'HIGH': Range > 1400, high probability of overflow/underflow
    
    See Also
    --------
    analyzeNumericalPrecision : Comprehensive precision analysis with detailed diagnostics
    numpy.ptp : Peak-to-peak (range) calculation used internally
    """
    # Convert input to numpy array for efficient range calculation
    likelihoods = np.asarray(likelihoods)
    
    # Calculate the range (peak-to-peak) of log-likelihood values
    # This represents the maximum possible pairwise difference magnitude
    likelihoodRange = np.max(likelihoods) - np.min(likelihoods)
    
    # Classify risk level based on likelihood range thresholds
    if likelihoodRange > HIGH_RISK_LIKELIHOOD_RANGE:
        # Range > 1400 guarantees some pairwise differences exceed ±700
        return "HIGH"
    elif likelihoodRange > MODERATE_RISK_LIKELIHOOD_RANGE:
        # Range 700-1400 may produce problematic differences depending on distribution
        return "MODERATE"
    else:
        # Range ≤ 700 cannot produce differences exceeding ±700
        return "LOW"


