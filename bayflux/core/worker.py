"""
Worker Module for BayFlux Parallel MCMC Sampling.

This module contains worker functions for multiprocessing-based Bayesian flux
analysis using shared memory for efficient inter-process communication.
"""

import time
import logging
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from bayflux.core.sampler import MCMCACHRSampler


def samplerSharedMemory(
    workerId: int,
    seed: int,
    model: 'bayflux.ReactionNetwork',
    center: np.ndarray,
    logLikelihood: callable,
    positionShmName: str,
    proposalsShmName: str,
    likelihoodsShmName: str,
    statusShmName: str,
    fluxVectorSize: int,
    numProposals: int,
    lock: mp.Lock
):
    """
    Worker process function for parallel MCMC sampling using shared memory.

    Each worker attaches to shared memory and continuously generates new
    proposals, calculates their likelihoods, and writes the results back
    to shared memory until terminated by the parent process.

    Parameters
    ----------
    workerId : int
        A unique identifier for the worker process (1-indexed).
    seed : int
        The random seed for this worker to ensure reproducible sampling.
    model : bayflux.ReactionNetwork
        The metabolic model object used for constraint-based sampling.
    center : np.ndarray
        The center point of the flux space for the ACHR sampler, used as
        the reference point for hit-and-run sampling.
    logLikelihood : callable
        A pre-configured function to compute the log-likelihood of a flux
        vector against experimental data.
    positionShmName : str
        The name of the shared memory block containing the current position
        (flux vector) being explored.
    proposalsShmName : str
        The name of the shared memory block for storing all proposals
        (current position + new proposals from workers).
    likelihoodsShmName : str
        The name of the shared memory block for storing likelihood values
        corresponding to each proposal.
    statusShmName : str
        The name of the shared memory block for tracking worker status
        (0=waiting, 1=ready to process, 2=processed).
    fluxVectorSize : int
        The dimensionality of the flux vector (number of reactions).
    numProposals : int
        The total number of parallel proposals being generated.
    lock : mp.Lock
        A multiprocessing lock for synchronizing access to shared memory
        between workers and the main process.

    Returns
    -------
    None
        The function operates entirely through side effects on shared memory
        and continues until the process is terminated by the parent.

    Notes
    -----
    This function implements the worker logic for parallel Bayesian flux
    analysis where multiple workers generate proposals simultaneously using
    ACHR sampling. The shared memory architecture allows for efficient
    communication without expensive serialization/deserialization overhead.

    The worker follows this cycle:
    1. Wait for signal from main process (status = 1)
    2. Copy current position from shared memory
    3. Generate new proposal using ACHR sampling
    4. Calculate likelihood of the proposal
    5. Write results back to shared memory
    6. Signal completion (status = 2)
    7. Repeat until terminated

    Error Handling
    --------------
    If an error occurs during sampling or likelihood calculation, the worker
    writes -inf as the likelihood value and marks itself as processed to
    prevent deadlock.
    """
    # Attach to existing shared memory blocks created by the main process
    positionShm = shared_memory.SharedMemory(name=positionShmName)
    positionShared = np.ndarray((fluxVectorSize,), dtype=np.float64, buffer=positionShm.buf)
    
    proposalsShm = shared_memory.SharedMemory(name=proposalsShmName)
    proposalsShared = np.ndarray((numProposals + 1, fluxVectorSize), dtype=np.float64, buffer=proposalsShm.buf)
    
    likelihoodsShm = shared_memory.SharedMemory(name=likelihoodsShmName)
    likelihoodsShared = np.ndarray((numProposals + 1,), dtype=np.float64, buffer=likelihoodsShm.buf)
    
    statusShm = shared_memory.SharedMemory(name=statusShmName)
    statusShared = np.ndarray((numProposals + 1,), dtype=np.int32, buffer=statusShm.buf)
    
    # Initialize the MCMC-ACHR sampler for this worker
    # Each worker maintains its own sampler instance with the same center
    mcmcAchrSampler = MCMCACHRSampler(model, thinning=100, seed=seed)
    mcmcAchrSampler.center = center

    # Main worker loop - continues until process is terminated by parent
    while True:
        # Wait for signal from main process to start processing
        while True:
            with lock:
                if statusShared[workerId] == 1:  # Status 1 = ready to process
                    break
            time.sleep(0.001)  # Small sleep to prevent excessive CPU usage
        
        try:
            # Safely copy the current position from shared memory
            with lock:
                currentPosition = positionShared.copy()
                # Update sampler's previous position to current shared position
                mcmcAchrSampler.prev = currentPosition
            
            # Generate new proposal using ACHR sampling algorithm
            # lockCenter=True: keeps the center fixed during sampling
            # validatecheck=True: performs constraint validation on the sample
            mcmcAchrSampler._MCMCACHRSampler__single_iteration(
                lockCenter=True, 
                validatecheck=True
            )
            
            # Calculate log-likelihood of the newly generated proposal
            proposalLikelihood = logLikelihood(mcmcAchrSampler.prev)
            
            # Write results back to shared memory atomically
            with lock:
                # Store the new proposal in the designated slot for this worker
                proposalsShared[workerId] = mcmcAchrSampler.prev
                # Store the corresponding likelihood value
                likelihoodsShared[workerId] = proposalLikelihood
                # Mark this worker as completed (status 2 = processed)
                statusShared[workerId] = 2
                
        except Exception as e:
            # Log the error for debugging purposes
            logging.error(f"Worker {workerId} encountered error during sampling: {e}")
            
            # Set likelihood to negative infinity to indicate failure
            # This ensures the proposal will have minimal acceptance probability
            with lock:
                likelihoodsShared[workerId] = float('-inf')
                # Still mark as processed to prevent deadlock in main process
                statusShared[workerId] = 2
    
    