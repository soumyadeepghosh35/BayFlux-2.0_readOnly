"""
BayFlux v2.0 Main Module.

This module implements the complete workflow for BayFlux - Bayesian method to quantify 
metabolic Fluxes and their uncertainty at the genome scale. It describes the entire 
process of metabolic modeling and Multiple-Proposal Markov Chain Monte Carlo (MP-MCMC) sampling using 
multiprocessing with shared memory architecture for efficient parallel computation.

The module handles:
- Configuration file parsing and validation
- Metabolic model loading and preprocessing using COBRA
- Elementary Metabolite Unit (EMU) network compilation
- Parallel MCMC sampling with Artificial Centering Hit-and-Run (ACHR) sampling
- Shared memory management for inter-process communication
- Results collection and output generation
- Performance monitoring and error handling (in developer version only)

Key Features:
- Multi-process parallel sampling for improved performance
- Shared memory architecture to minimize serialization overhead
- Robust error handling and numerical stability measures (in developer version only)
- Comprehensive timing and resource monitoring (in developer version only)
- Resume capability for long-running computations (in developer version only)

Dependencies:
- BayFlux core modules for metabolic modeling and sampling
- COBRApy for metabolic network manipulation
- NumPy/SciPy for numerical computations
- Multiprocessing for parallel execution
- YAML for configuration management

Author: BayFlux Development Team
License: See license.txt
Version: 2.0
"""

# Import standalone python libraries
import os
import sys
import ntpath
import re
import argparse
import yaml
import multiprocessing as mp
import datetime
import time
from functools import partial
from multiprocessing import shared_memory, Lock
from typing import Optional, Dict, List, Tuple, Any, Union
from tabulate import tabulate
import numpy as np
from threadpoolctl import ThreadpoolController
import csv  # (in developer version only)
import logging  # (in developer version only)
from line_profiler import LineProfiler  # (in developer version only)

# BayFlux specific imports
import cobra
#cobra.Configuration().solver = "glpk"
import bayflux
from bayflux.core.transitionProbabilities import calculateTransitionProbabilitiesBayflux
from bayflux.core.steadyState import steadyStateDistribution
from bayflux.core.sampler import MCMCACHRSampler
from bayflux.core.acceptanceRate import computeAcceptanceRate
from bayflux.core.likelihood import logLikelihood, normpdf
from bayflux.core.worker import samplerSharedMemory
from bayflux.core.utilities import logSystemResources, analyzeNumericalPrecision, quickPrecisionCheck

# Global constants for thread and numerical library control
CONTROLLER = ThreadpoolController()

# Environment variable configuration to control numerical library threading
# These settings ensure predictable behavior in multiprocessing environments
# by limiting each process to single-threaded numerical operations
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'

# Timing constants for performance measurement
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60

# Output formatting constants
PACKAGE_INFO_PADDING = 2
SEPARATOR_WIDTH = 45
BOX_SEPARATOR_WIDTH = 60


#@profile  # Line profiler decorator - kept for debugging purposes only
@CONTROLLER.wrap(limits=1, user_api='blas')
def main() -> None:
    """
    Main function for BayFlux: This function coordinates the complete end-to-end process of 
    Bayesian metabolic flux analysis, including configuration parsing, model setup, 
    parallel MCMC sampling and results generation.

    The workflow consists of several key phases:
    1. Configuration and input parameter processing
    2. Metabolic model loading and validation  
    3. EMU network compilation for isotope simulation
    4. Warmup sampling to establish chain center
    5. Parallel MP-MCMC sampling with shared memory
    6. Results collection and output generation
    7. Performance reporting and cleanup

    Returns
    -------
    None
        The function operates through saving final simulation results to files 
        specified in the configuration and printing execution summaries to console.

    Raises
    ------
    SystemExit
        If critical errors occur during configuration parsing, model loading,
        or other essential setup operations.
    ValueError
        If configuration parameters are invalid or inconsistent.
    FileNotFoundError
        If required input files (model, configuration, data) cannot be found.
    
    Notes
    -----
    The function is decorated with `@CONTROLLER.wrap(limits=1, user_api='blas')` 
    to limit the number of threads used by BLAS libraries, preventing 
    oversubscription in the multiprocessing environment.

    The function uses shared memory for efficient inter-process communication
    during parallel sampling, which significantly reduces overhead compared
    to traditional pickling-based approaches.

    Examples
    --------
    Run from command line with configuration file:
    
    >>> bayflux config.yaml > bayflux.out
    
    The configuration file should contain all necessary parameters for
    the analysis including model files, data files, and sampling parameters.
    """
    startTotalCodeExecutionTime = time.time()

    # Display package information and citation details in formatted box
    lines = [
        "Package: BayFlux (https://github.com/JBEI/bayflux)",
        "",
        "Please cite:",
        "1. ",
        "2. BayFlux: A Bayesian method to quantify metabolic",
        "Fluxes and their uncertainty at the genome scale, Backman TWH et al.,",
        "PLOS Computational Biology 19, 11 (2023)",
        "https://doi.org/10.1371/journal.pcbi.1011111"
    ]

    # Display package information to console
    maxLength = max(len(line) for line in lines)
    boxWidth = maxLength + (PACKAGE_INFO_PADDING * 2)

    print("┌" + "─" * boxWidth + "┐")
    for line in lines:
        paddingNeeded = maxLength - len(line)
        print("│" + " " * PACKAGE_INFO_PADDING + line + " " * (PACKAGE_INFO_PADDING + paddingNeeded) + "│")
    print("└" + "─" * boxWidth + "┘")

    print("-" * SEPARATOR_WIDTH)
    print("!! Job started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "GMT !!")
    print("-" * SEPARATOR_WIDTH)

    startReadingInputParamTime = time.time()
    
    # Parse command-line arguments for configuration file
    parser = argparse.ArgumentParser(description='BayFlux command line version')
    parser.add_argument('config', help='configuration file (yaml format)')
    args = parser.parse_args()

    # Load and parse YAML configuration file
    with open(args.config, 'r') as stream:
        try:
            configData = yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print(f"Error parsing YAML configuration: {exception}")
            sys.exit(1)

    # Extract configuration parameters into local variables for easier access
    labelingDataFile = configData['mdvFile']
    thinning = configData['thinning']
    warmupSamplesPerTask = configData['centeringSamplesPerTask']
    samplesPerTask = configData['bayesianSamplesPerTask']
    outputBaseName = configData['outputBaseName']
    averageError = configData['averageError']
    seed = configData['seed']
    numProposals = configData['numProposals']
    subsamplesPerIteration = configData['subsamplesPerIteration']
    maxFlux = configData['maxFlux']

    # Display(print) input parameters for verification
    print("##### Input parameters #####")
    print("Warmup samples:", warmupSamplesPerTask)
    print("No. of Bayesian iteration:", samplesPerTask)
    print("No. of parallel proposals (N):", numProposals)
    print("Sub-samples per iteration (M):", subsamplesPerIteration)
    print("Thinning:", thinning)
    print(f"Total samples (Thinning x Bayesian iteration x Parallel proposals):{thinning * samplesPerTask * numProposals}")
    print(f"Flux space bound: ±{maxFlux}\n")

    print("##### Metabolic model information #####")
    
    # Load metabolic model from SBML file 
    try:
        cobraPyModel = cobra.io.read_sbml_model(configData['modelFile'])
    except Exception as e:
        print(f"Error reading SBML model: {e}")
        sys.exit(1)
    
    # Convert COBRApy model to BayFlux ReactionNetwork object
    model = bayflux.ReactionNetwork(cobraPyModel)
    print("Model Name:", model.name)
    print("Number of Reactions:", len(model.reactions))
    print("Number of Metabolites:", len(model.metabolites))
    print("Number of Genes:", len(model.genes))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEBUGGING SECTION: Model stability diagnosis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following commented section would be used for model stability analysis
    # This is kept for future debugging and development purposes
    # is_stable = diagnoseModelStability(model, config_file)
    # if not is_stable:
    #     print(" Model appears unstable. Aborting sampling.")
    #     return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n##### Job specific information #####")
    
    # Create output directory if it doesn't exist
    outputFolder = ntpath.dirname(configData['outputBaseName'])
    try:
        os.mkdir(outputFolder)
    except FileExistsError:
        pass  # Directory already exists, which is fine

    # Execute substrate labeling dictionary code from external file
    # This approach allows for flexible substrate labeling definitions
    localVars = locals()
    try:
        exec(open(configData['substrateLabelingDict']).read(), globals(), localVars)
    except Exception as e:
        print(f"Error executing substrate labeling file: {e}")
        sys.exit(1)
    
    substrateLabelingDict = localVars["substrateLabelingDict"]
    assert isinstance(substrateLabelingDict, dict), 'substrateLabelingDict file did not contain a python dict named substrateLabelingDict as required!'

    # Identify existing output files for potential resume functionality
    files = os.listdir()
    matchingFiles = list(filter(lambda x: re.match(r'^' + outputBaseName + r'(\d{3})\.npy$', x), files))
    fileNameIntegers = [int(re.sub(r'^' + outputBaseName + r'(\d{3})\.npy', r'\1', filename)) for filename in matchingFiles]

    def integerToFilename(outputBaseName: str, x: int, zfill: int = 3) -> str:
        """
        Generate a standardized output filename from a base name and integer.

        This utility function creates consistent file naming for sequential output 
        files, using zero-padding to ensure proper alphabetical sorting of files.

        Parameters
        ----------
        outputBaseName : str
            Base name for the output file without extension.
        x : int
            Integer identifier to append to the filename.
        zfill : int, optional
            Number of digits for zero-padding the integer, by default 3.
            This ensures files are named like file001.npy, file002.npy, etc.

        Returns
        -------
        str
            Formatted filename with zero-padded integer and .npy extension.

        """
        return f"{outputBaseName}{str(x).zfill(zfill)}.npy"

    # Determine file names for resuming and new output
    if len(fileNameIntegers) > 0:
        previousFileName = integerToFilename(outputBaseName, max(fileNameIntegers))
        newFileName = integerToFilename(outputBaseName, max(fileNameIntegers) + 1)
        print(f'Resuming from previous instance {len(fileNameIntegers)}')
    else:
        print('New run, not resuming')
        previousFileName = None
        newFileName = integerToFilename(outputBaseName, 1)

    # Apply flux constraints from configuration file
    model.readFluxConstraints(configData['fluxBoundsFile'])

    # Set maximum flux bounds to prevent unrealistic flux values
    limit = configData['maxFlux']
    for reaction in model.reactions:
        if reaction.lower_bound < -limit:
            reaction.lower_bound = -limit
        if reaction.upper_bound > limit:
            reaction.upper_bound = limit

    # Validate model functionality with Flux Balance Analysis (FBA)
    model.optimize()

    # Load atom transition mappings for isotope tracking
    model.readAtomTransitions(configData['transitionsFile'])

    # Import experimental mass distribution data
    importedMDVs = bayflux.readMassDistribution(model, configData['mdvFile'])

    endReadingInputParamTime = time.time()
    readingInputParamTime = endReadingInputParamTime - startReadingInputParamTime

    startCompileEmuNetworkTime = time.time()
    
    # Pre-compile EMU network for efficient isotope simulation
    print("Pre-compile EMU network")
    compiledData = bayflux.emuCompile(list(importedMDVs.data.keys()), model, substrateLabelingDict)

    # Create configured log-likelihood function with fixed parameters
    # This partial function application improves performance by avoiding repeated parameter passing
    configuredLogLikelihood = partial(
        logLikelihood,
        lcmsResults=importedMDVs,
        compiledData=compiledData,
        substrateLabelingDict=substrateLabelingDict,
        errorSD=configData['averageError']
    )

    endCompileEmuNetworkTime = time.time()
    compileEmuNetworkTime = endCompileEmuNetworkTime - startCompileEmuNetworkTime

    startWarmupSamplingTime = time.time()
    
    # Initialize MCMC-ACHR sampler with specified parameters
    mcmcAchrSampler = MCMCACHRSampler(model, thinning=100, seed=seed)

    # Define filename for storing the computed center point
    centerFile = f"{outputBaseName}_center.npy"

    # Load existing center or compute new one
    if os.path.isfile(centerFile):
        # Resume from previously computed center to save time
        print(f'Loading previously saved center from file: {centerFile}')
        mcmcAchrSampler.center = np.load(centerFile)
    else:
        # Compute new center using uniform sampling across feasible space
        print('Collecting uniform samples to get the center ...')
        mcmcAchrSampler.sample(warmupSamplesPerTask, fluxes=False, validatecheck=True)
        np.save(centerFile, mcmcAchrSampler.center, allow_pickle=False)

    endWarmupSamplingTime = time.time()
    warmupSamplingTime = endWarmupSamplingTime - startWarmupSamplingTime

    startMpMcmcSamplingTime = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEBUGGING SECTION: Initial position analysis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following commented sections are preserved for debugging purposes
    # These would check the distance from center to initial position and log detailed information
    # distance_from_center = np.linalg.norm(position - mcmcAchrSampler.center)
    # print(f"Distance from center to initial position: {distance_from_center:.6f}")
    # if distance_from_center < 1e-10:
    #     print("MP-MCMC sampling starts from center")
    # else:
    #     print(f"MP-MCMC sampling starts from different position (distance: {distance_from_center:.6f})")

    # DEBUGGING SECTION: CSV logging setup
    # This section would set up CSV logging for detailed proposal tracking
    # logLikehood_csv_writer = csv.writer(open('mcmc_proposals_log.csv', 'a', newline=''))
    # logLikehood_csv_writer.writerow(['Iteration', 'Proposal_No', 'Likelihood', 'Proposal_Type'])

    # DEBUGGING SECTION: Initial state logging
    # These lines would log the initial center and position information
    # print(f"Center: [{', '.join([f'{val:.6f}' for val in mcmcAchrSampler.center[:5]])}...], logLikelihood: {logLikelihood(mcmcAchrSampler.center)}")
    # print(f"Initial position: [{', '.join([f'{val:.6f}' for val in position[:5]])}...], logLikelihood: {logLikelihood(position)}")
    # print(f"Initial state: {position}, Loglikelihood:{likelihood_of_position}")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize parallel MP-MCMC sampling
    print(f'Spawning {numProposals} parallel samplers')

    # Shared Memory Architecture Setup
    # This section creates shared memory blocks for efficient inter-process communication
    # - positionShared: Current flux vector being explored
    # - proposalsShared: All proposals (current + new proposals from workers)  
    # - likelihoodsShared: Likelihood values for each proposal
    # - statusShared: Worker process status tracking (0=waiting, 1=ready, 2=processed)

    fluxVectorSize = len(model.variables)

    # Create shared memory block for current position (flux vector)
    positionShm = shared_memory.SharedMemory(create=True, size=fluxVectorSize * np.float64().nbytes)
    positionShared = np.ndarray((fluxVectorSize,), dtype=np.float64, buffer=positionShm.buf)

    # Create shared memory for proposals (numProposals + 1 to include current position)
    proposalsShm = shared_memory.SharedMemory(create=True, size=(numProposals + 1) * fluxVectorSize * np.float64().nbytes)
    proposalsShared = np.ndarray((numProposals + 1, fluxVectorSize), dtype=np.float64, buffer=proposalsShm.buf)

    # Create shared memory for likelihood values
    likelihoodsShm = shared_memory.SharedMemory(create=True, size=(numProposals + 1) * np.float64().nbytes)
    likelihoodsShared = np.ndarray((numProposals + 1,), dtype=np.float64, buffer=likelihoodsShm.buf)

    # Create shared memory for worker status flags
    statusShm = shared_memory.SharedMemory(create=True, size=(numProposals + 1) * np.int32().nbytes)
    statusShared = np.ndarray((numProposals + 1,), dtype=np.int32, buffer=statusShm.buf)
    statusShared.fill(0)  # Initialize all workers to waiting status

    # Initialize synchronization lock and starting position
    lock = Lock()
    position = mcmcAchrSampler.prev
    positionShared[:] = position
    accumulatedAcceptanceRates = []

    # Calculate likelihood of initial position and store in shared memory
    likelihoodOfPosition = configuredLogLikelihood(position)
    likelihoodsShared[0] = likelihoodOfPosition
    proposalsShared[0] = position

    # Launch worker processes for parallel proposal generation
    childProcesses = []
    for workerId in range(numProposals):
        # Generate unique seed for each worker to ensure diversity
        thisSeed = (seed + (seed * (workerId + 1)))
        process = mp.Process(
            target=samplerSharedMemory,
            args=(
                workerId + 1,  # workerId + 1 because index 0 is reserved for current position
                thisSeed,
                model,
                mcmcAchrSampler.center,
                configuredLogLikelihood,
                positionShm.name,
                proposalsShm.name,
                likelihoodsShm.name,
                statusShm.name,
                fluxVectorSize,
                numProposals,
                lock
            )
        )
        process.start()
        childProcesses.append(process)

    # Initialize results storage array
    data = np.zeros(shape=(samplesPerTask, len(model.variables)), dtype=np.float64, order='C')
    print(f"Shape of initialized result dimensions: {data.shape}\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEBUGGING SECTION: Additional counters and tracking variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These commented sections preserve debugging infrastructure for development
    # saved_samples_counter = 0
    # accumulated_acceptance_rates = []
    # acceptance_count_serialLike = 0
    # total_proposals_serialLike = 0

    # DEBUGGING SECTION: Numerical precision tracking
    # This system would track the frequency of different numerical precision issues
    # precision_counters = {
    #     "HIGH": 0,
    #     "MODERATE": 0,
    #     "LOW": 0
    # }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Initialize error tracking for production monitoring
    errorCounters = {
        "Negative Values in Transition Matrix": 0,
        "Rows Not Summing to 1": 0,
        "Numerical Precision Issues": 0,
    }
    totalSkippedIterations = 0

    # Main MP-MCMC Sampling Loop
    for sampleNumber in range(samplesPerTask * thinning):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # DEBUGGING SECTION: Progress tracking
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # percentage_of_completed_iteration = ((sampleNumber+1)/(samplesPerTask * thinning))*100
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Synchronize worker processes and update shared memory
        with lock:
            positionShared[:] = position
            proposalsShared[0] = position  # Store current position as proposal 0
            likelihoodsShared[0] = likelihoodOfPosition  # Store current position's likelihood
            statusShared[1:] = 1  # Signal all workers to start processing
            statusShared[0] = 0  # Current position is not being processed by workers

        # Wait for all worker processes to complete their proposals
        allDone = False
        while not allDone:
            with lock:
                allDone = np.all(statusShared[1:] == 2)  # Check if all workers are done (status = 2)
            if not allDone:
                time.sleep(0.001)  # Small sleep to prevent excessive CPU usage

        # Process completed proposals with error handling and retry logic
        retry = True
        while retry:
            try:
                # Copy proposals and likelihoods from shared memory to local arrays
                samplesTempNp = np.asarray(proposalsShared)
                likelihoodsNp = np.asarray(likelihoodsShared).reshape(-1, 1)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # DEBUGGING SECTION: Numerical precision analysis
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # This extensive debugging system analyzes numerical stability of proposals
                # risk = quickPrecisionCheck(likelihoodsShared)
                # precision_counters[risk] += 1
                # if risk == "HIGH":
                #     try:
                #         analysis_results = analyzeNumericalPrecision(samplesTempNp, likelihoodsShared, numProposals)
                #     except Exception as e:
                #         print(f"Analysis failed: {e}")
                # if risk == "HIGH":
                #     transition_prob_func = calculate_transition_probabilities_log_space
                # elif risk == "MODERATE":
                #     transition_prob_func = calculate_transition_probabilities_stable
                # else:
                #     transition_prob_func = calculate_transition_probabilities

                # DEBUGGING SECTION: Detailed proposal logging
                # These sections would log detailed information about each proposal
                # print(f"\n----- Iteration {sampleNumber + 1} -----")
                # print("Current position (proposal 0):", samplesTempNp[0][:5], "...", f"Likelihood: {likelihoodsNp[0][0]:.6f}")
                # for i in range(1, numProposals + 1):
                #     print(f"Proposal {i}: {samplesTempNp[i][:5]} (len: {len(samplesTempNp[i])}), Likelihood: {likelihoodsNp[i][0]:.6f}")
                # iteration_num = sampleNumber + 1
                # logLikehood_csv_writer.writerow([iteration_num, 0, f"{likelihoodsNp[0][0]:.6f}", "Current"])
                # for i in range(1, numProposals + 1):
                #     logLikehood_csv_writer.writerow([iteration_num, i, f"{likelihoodsNp[i][0]:.6f}", "Proposal"])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Compute transition probability matrix using ACHR-aware asymmetric Generalized Metropolis-Hastings
                transitionProbabilityMatrix = calculateTransitionProbabilitiesBayflux(samplesTempNp, likelihoodsShared, numProposals)

                # Compute steady-state distribution from transition matrix
                computedSteadyStateDistribution = steadyStateDistribution(transitionProbabilityMatrix)
                normalizedDistribution = computedSteadyStateDistribution / np.sum(computedSteadyStateDistribution)
                
                # Validate the computed distribution for numerical stability
                if not np.all(normalizedDistribution >= 0):
                    raise ValueError("Normalized distribution has negative values")
                if not np.isclose(normalizedDistribution.sum(), 1):
                    raise ValueError("Normalized distribution does not sum to 1")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # DEBUGGING SECTION: Detailed acceptance probability logging
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # This section would log acceptance probabilities for each proposal
                # output_parts = [f"--> Iteration {sampleNumber + 1}:"]
                # for i in range(len(samplesTempNp)):
                #     sample_val = samplesTempNp[i, 0]
                #     likelihood = likelihoodsNp[i, 0]
                #     norm_prob = normalizedDistribution[i]
                #     if i==0:
                #         output_parts.append(f" Proposal {i} (old): logAcceptProbability={np.log(norm_prob):.6f} | acceptProbability={norm_prob:.6f}")
                #     else:
                #         output_parts.append(f" Proposal {i+1} (new): logAcceptProbability={np.log(norm_prob):.6f} | acceptProbability={norm_prob:.6f}")
                # print("".join(output_parts))
                # print("\nAccepted samples:")
                # for idx, sample_idx in enumerate(newSamplesIndices):
                #     print(f"Sample {idx+1}: Proposal {sample_idx} with likelihood {likelihoodsNp[sample_idx][0]:.6f} (Weight: {normalizedDistribution[sample_idx]:.6f})")
                # print(f"Iteration {sampleNumber + 1} summary: Next position will be from proposal {indexOfPosition} with likelihood {likelihoodOfPosition:.6f}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # Subsample M samples based on the computed steady-state distribution
                newSamplesIndices = np.random.choice(len(samplesTempNp), size=subsamplesPerIteration, p=normalizedDistribution, replace=True)
                newSamples = samplesTempNp[newSamplesIndices]

                # Update current position to the last selected sample for next iteration
                position = newSamples[-1].copy()
                indexOfPosition = newSamplesIndices[-1]
                likelihoodOfPosition = likelihoodsNp[indexOfPosition][0]

                # Update shared memory with new position for workers
                with lock:
                    positionShared[:] = position

                retry = False  # Exit retry loop if no errors occurred
            
            except ValueError as e:
                print(f"Skipping iteration {sampleNumber + 1} due to error: {e}")
                totalSkippedIterations += 1

                # Increment specific error counters based on error message for monitoring
                if "Negative distribution has negative values" in str(e):
                    errorCounters["Negative Values in Transition Matrix"] += 1
                elif "Transition matrix rows do not sum to 1" in str(e):
                    errorCounters["Rows Not Summing to 1"] += 1
                elif "Power iteration did not converge" in str(e):
                    errorCounters["Numerical Precision Issues"] += 1
                else:
                    print(f"Unknown error occurred: {e}")

                retry = True  # Retry sampling from the last accepted sample

        # Store sample in results array only at thinning intervals
        if (sampleNumber + 1) % thinning == 0:
            dataIndex = (sampleNumber // thinning)

            # Validate array bounds to prevent memory corruption
            if dataIndex >= data.shape[0]:
                raise ValueError("Attempt to write beyond allocated space of data array.")

            data[dataIndex, :] = position

       
    # Calculate acceptance rate of samples
    acceptanceRate = computeAcceptanceRate(newSamples, burnIn=0)
    #print(f"Accceptance rate: {acceptanceRate:.2f}%")
    accumulatedAcceptanceRates.append(acceptanceRate)

    
    # Cleanup: Terminate all worker processes gracefully
    for process in childProcesses:
        process.terminate()
        process.join()

    # Cleanup: Release shared memory resources
    positionShm.close()
    positionShm.unlink()
    proposalsShm.close()
    proposalsShm.unlink()
    likelihoodsShm.close()
    likelihoodsShm.unlink()
    statusShm.close()
    statusShm.unlink()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEBUGGING SECTION: Numerical precision summary reporting
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This extensive section would provide detailed analysis of numerical stability
    # print("\n##### Numerical Precision Summary #####")
    # total_iterations = samplesPerTask * thinning
    # for precision_level, count in precision_counters.items():
    #     percentage = (count / total_iterations) * 100
    #     print(f"---> {precision_level} precision risk: {count}/{total_iterations} iterations ({percentage:.2f}%)")
    # print(f"\n---> Methods used:")
    # print(f"    Log-space method: {precision_counters['HIGH']}/{total_iterations} iterations ({(precision_counters['HIGH']/total_iterations)*100:.2f}%)")
    # print(f"    Stable method: {precision_counters['MODERATE']}/{total_iterations} iterations ({(precision_counters['MODERATE']/total_iterations)*100:.2f}%)")
    # print(f"    Original method: {precision_counters['LOW']}/{total_iterations} iterations ({(precision_counters['LOW']/total_iterations)*100:.2f}%)")
    # max_count = max(precision_counters.values())
    # most_frequent_risk = [risk for risk, count in precision_counters.items() if count == max_count][0]
    # if most_frequent_risk == "HIGH":
    #     primary_method = "Log-space (HIGH risk protection)"
    # elif most_frequent_risk == "MODERATE":
    #     primary_method = "Stable (MODERATE risk protection)"
    # else:
    #     primary_method = "Original (LOW risk - optimal performance)"
    # print(f"---> Most frequently used method: {primary_method}")
    # logSystemResources()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Report execution error information for production monitoring
    print("\n##### Job execution error information #####")
    print(f"---> Total skipped iterations: {totalSkippedIterations} ({(totalSkippedIterations / (samplesPerTask * thinning)) * 100:.2f}%)")
    
    # Detailed breakdown of error types encountered during sampling
    for errorType, count in errorCounters.items():
        percentage = (count / (samplesPerTask * thinning)) * 100
        print(f"{errorType}: {count} ({percentage:.2f}%)")

    print("\n##### Results information #####")
    

    # This section would compute and report acceptance rates if available
    if numProposals != 1:
        meanAcceptanceRate = np.mean(accumulatedAcceptanceRates)
        print(f"---> Acceptance rate: {meanAcceptanceRate:.2f}%")
    
    print(f"---> Result dimensions: {data.shape}")
    print(f"---> Results saved to file: {newFileName}")

    # Save results to numpy binary format for efficient storage
    np.save(newFileName, data, allow_pickle=False)
    
    endMpMcmcSamplingTime = time.time()
    totalMpMcmcSamplingTime = endMpMcmcSamplingTime - startMpMcmcSamplingTime
    
    # Convert sampling time to human-readable format
    days, remainder = divmod(totalMpMcmcSamplingTime, SECONDS_PER_DAY)
    hours, remainder = divmod(remainder, SECONDS_PER_HOUR)
    minutes, seconds = divmod(remainder, SECONDS_PER_MINUTE)
    days, hours, minutes, seconds = int(days), int(hours), int(minutes), int(seconds)

    # Calculate total execution time and convert to human-readable format
    endTotalCodeExecutionTime = time.time()
    totalCodeExecutionTime = endTotalCodeExecutionTime - startTotalCodeExecutionTime
    days, remainder = divmod(totalCodeExecutionTime, SECONDS_PER_DAY)
    hours, remainder = divmod(remainder, SECONDS_PER_HOUR)
    minutes, seconds = divmod(remainder, SECONDS_PER_MINUTE)
    days, hours, minutes, seconds = int(days), int(hours), int(minutes), int(seconds)
    
    print("\n##### Timing information #####")
    print(f"---> Total execution time: {days:02d} Day:{hours:02d} Hr:{minutes:02d} Min:{seconds:02d} Sec ({totalCodeExecutionTime:.2f} sec)")

    # Prepare detailed timing breakdown for performance analysis
    timingData = [
        ["Reading input parameters", f"{(readingInputParamTime/totalCodeExecutionTime)*100:.2f}%"],
        ["EMU network decomposition", f"{(compileEmuNetworkTime/totalCodeExecutionTime)*100:.2f}%"],
        ["Warmup sampling", f"{(warmupSamplingTime/totalCodeExecutionTime)*100:.2f}%"],
        ["MP-MCMC sampling", f"{(totalMpMcmcSamplingTime/totalCodeExecutionTime)*100:.2f}%"]
    ]

    headers = ["Operation", "%(Total-Time)"]

    print("---> Timing breakdown:")
    print(tabulate(timingData, headers=headers, tablefmt="grid"))

    # Calculate and report performance metrics
    samplesPerSecond = (thinning * samplesPerTask * numProposals) / totalMpMcmcSamplingTime
    print(f"---> Number of samples executed per second: {samplesPerSecond:.2f}")
    
    # Calculate MP-MCMC iterations per second for performance monitoring
    mpMcmcIterationsPerSecond = (samplesPerTask * thinning) / totalMpMcmcSamplingTime
    print(f"---> Number of MP-MCMC iterations per second: {mpMcmcIterationsPerSecond:.2f}")
    
    print("-" * BOX_SEPARATOR_WIDTH)
    print("!! Job successfully completed at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "GMT !!")
    print("-" * BOX_SEPARATOR_WIDTH)



if __name__ == "__main__":
    main()