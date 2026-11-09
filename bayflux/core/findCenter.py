#!/usr/bin/env python3

# This script is designed for running the BayFlux v2.0 genome-scale metabolic modeling framework.
# It reads a genome-scale metabolic model and sets up an MCMC (Markov Chain Monte Carlo) sampling process
# to calculate center points for the model, which are essential for further computational analysis.
# The script takes configuration parameters from a YAML file, which include paths to the input model,
# output file naming conventions, and various sampling parameters.
#
# Major Steps:
# 1. Load the metabolic model from an SBML file.
# 2. Initialize the MCMC sampler with user-defined parameters (e.g., thinning interval, random seed).
# 3. Execute a sampling process to compute 'center' points that represent intermediate states.
# 4. Save the computed center points to .npy files for further use in convergence analysis or as input
#    for additional BayFlux calculations.
#
# The results are saved in two forms:
# - The last center point is saved separately for immediate use in BayFlux analysis.
# - All center points, thinned according to the provided interval, are saved for checking convergence
#   and ensuring model accuracy.



import os
import sys

if os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(sys.path[0]):
    sys.path.pop(0)
import bayflux
import numpy as np
import yaml
import cobra
from bayflux.core.sampler import MCMCACHRSampler
import argparse
import time
import pandas as pd
import ntpath

# Set environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


#@profile
def autoCenter(sampler, samples, centerPoints, confirmValidity=True):
    """
    Function to calculate centers for drift analysis.
    
    Parameters:
    - sampler: The MCMC sampler object
    - samples: Total number of samples to collect
    - centerPoints: Number of center points to collect for drift calculation
    - confirmValidity: Whether to check if samples are valid (default: True)
    
    Returns:
    - centers: A numpy array of center points collected over time
    """
    centers = np.zeros((centerPoints, sampler.center.shape[0]))
    samplesPerStep = int(samples / centerPoints)
    
    for step in range(centerPoints):
        samples = sampler.sample(samplesPerStep, fluxes=False, likelihood=False)
        
        # Confirm validity of samples, if required
        if confirmValidity:
            assert sum(sampler.validate(samples) == 'v') == samplesPerStep, 'Infeasible solutions'
        
        centers[step] = sampler.center  # Store the current center
    
    return centers

#@profile
def main():
    """Find the sampling center for a YAML config file.

    Saves the center, the samples used to find it and the center drift trace,
    then prints timing and array shapes.
    """
    # Start timing
    start_time = time.time()
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Calculate the center for MCMC sampling.")
    parser.add_argument("config_file", help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configuration file from command-line argument
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        try:
            configData = yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print(f"Error loading configuration file: {exception}")
            exit(1)

    # create output folder if it doesn't exist
    outputFolder = ntpath.dirname(configData['outputBaseName'])
    try:
        os.mkdir(outputFolder)
    except FileExistsError:
        pass
    

    print("##### Metabolic model information #####")
    # Read the model file
    cobrapymodel = cobra.io.read_sbml_model(configData['modelFile'])
    model = bayflux.ReactionNetwork(cobrapymodel)
    print("Model Name:", model.name)
    print("Number of Reactions:", len(model.reactions))
    print("Number of Metabolites:", len(model.metabolites))
    print("Number of Genes:", len(model.genes))

    # Parameters for center calculation
    warmupSamplesPerTask = configData['centeringSamplesPerTask']
    thinning = configData.get('thinning', 100)
    seed = configData.get('seed', None)

    # File paths for output
    center_file = configData['outputBaseName'] + '_center.npy'
    center_data_file = configData['outputBaseName'] + '_centerData.npy'
    center_drift_file = configData['outputBaseName'] + '_centerDrift.npy'
    
    # Import and apply exchange fluxes from file
    model.readFluxConstraints(configData['fluxBoundsFile'])
    
    # set max flux
    limit = configData['maxFlux']
    for reaction in model.reactions:
        #print(f"ID: {reaction.id})
        if reaction.lower_bound < -limit:
            reaction.lower_bound = -limit
        if reaction.upper_bound > limit:
            reaction.upper_bound = limit

    # run FBA to confirm model works
    model.optimize()

    # Read atom transitions and apply to model
    model.readAtomTransitions(configData['transitionsFile'])
    
    # Read mass distribution data
    #importedMDVs = bayflux.readMassDistribution(model, configData['mdvFile'])
    
    #compiledData = bayflux.emuCompile(list(importedMDVs.data.keys()), model, substrateLabelingDict)
    
    # Initialize MCMC sampler
    mcmcachr = cobra.sampling.MCMCACHRSampler(model, thinning=100, seed=seed)
    
    

    print("##### Input parameters #####")
    print("seed:", seed)
    print("Thinning:", thinning)
    print("centeringSamplesPerTask:", warmupSamplesPerTask)
    print(f"Total samples: {warmupSamplesPerTask * thinning} ({(warmupSamplesPerTask * thinning) / 1e6:.2f} millions)")
    
    # Use `autoCenter` to collect drift centers
    print("Calculating drift centers...")
    drift_centers = autoCenter(sampler=mcmcachr, samples=warmupSamplesPerTask * thinning, centerPoints=warmupSamplesPerTask // thinning, confirmValidity=False)

    # Save drift centers for drift analysis
    np.save(center_drift_file, drift_centers, allow_pickle=False)
    print(f"Center drift data saved to {center_drift_file} for drift analysis")

    # Check if center file exists, otherwise calculate center
    if os.path.isfile(center_file):
        print(f'Loading previously saved center from {center_file}')
        mcmcachr.center = np.load(center_file)
    else:
        print('Calculating center by collecting uniform samples...')
        # Collect warm-up samples to calculate the center
        warmup_samples = mcmcachr.sample(warmupSamplesPerTask * thinning, fluxes=False, validatecheck=True)
        thinned_samples = warmup_samples[::thinning]  # Apply thinning interval to samples
        mcmcachr.center = warmup_samples.mean(axis=0)
        
        # Save the center and all samples used for finding the center
        np.save(center_file, mcmcachr.center, allow_pickle=False)
        np.save(center_data_file, thinned_samples, allow_pickle=False)
        print(f'Saved center to: {center_file}, use this file for BayFlux calculation')
        print(f'Saved center data to: {center_data_file}, use this file for checking convergence of centers')

   
    # End timing and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Load saved files to determine their shapes
    center_shape = np.load(center_file).shape
    center_data_shape = np.load(center_data_file).shape
    center_drift_shape = drift_centers.shape

    # Print timing information and shapes
    print(f"Shape of center file ({center_file}): {center_shape}")
    print(f"Shape of center data file ({center_data_file}): {center_data_shape}")
    print(f"Shape of center drift file ({center_drift_file}): {center_drift_shape}")
    #print(f"Center calculated: {mcmcachr.center}")

    print(f"##### Timing Information #####")
    print(f"---> Total execution time: {elapsed_time:.2f} seconds")
    print(f"---> Number of samples executed per second: {(warmupSamplesPerTask * thinning) / elapsed_time:.2f}")

# Run main if this script is executed directly
if __name__ == "__main__":
    main()




