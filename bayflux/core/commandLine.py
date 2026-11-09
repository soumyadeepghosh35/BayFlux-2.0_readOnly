 # BayFlux standalone command line version, this package searches the flux space with MCMC

# With Intel Distribution for Python restricting cores speeds things up ~2x
# Note: this needs to be defined before importing other libraries
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'

import cobra
#cobra.Configuration().solver = "glpk"
import ntpath
import bayflux
import pandas as pd
import numpy as np
import operator
import time
from mpi4py import MPI
#from cobra.sampling import MCMCACHRSampler
from bayflux.core.sampler import MCMCACHRSampler
from bayflux.core.likelihood import normpdf
from os import path
import re
import argparse
import yaml

def main():
    """Run one serial BayFlux chain from a YAML config file.

    Under MPI, rank 0 collects the samples from every rank and saves them.
    """
    # parse command line options
    cobra.Configuration().processes = 1
    parser = argparse.ArgumentParser(description='BayFlux standalone command line version')
    parser.add_argument('config', help='configuration file (yaml format)')
    args = parser.parse_args()

   # parse config file into variable configData
    with open(args.config, 'r') as stream:
        try:
            configData = yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print(exception)

    # parse out command line options
    labelingDataFile = configData['mdvFile']
    thinning = configData['thinning'] 
    warmupSamplesPerTask = configData['centeringSamplesPerTask'] 
    samplesPertask = configData['bayesianSamplesPerTask']
    outputBaseName = configData['outputBaseName']
    averageError = configData['averageError']
    seed = configData['seed']

    # Read the model file
    cobrapymodel = cobra.io.read_sbml_model(configData['modelFile'])
    model = bayflux.ReactionNetwork(cobrapymodel)
    model

    # create output folder if it doesn't exist
    outputFolder = ntpath.dirname(configData['outputBaseName'])
    try:
        os.mkdir(outputFolder)
    except FileExistsError:
        pass

    # execute substrate labeling dict code to get substrateLabelingDict
    localvars = locals()
    exec(open(configData['substrateLabelingDict']).read(), globals(), localvars)
    substrateLabelingDict = localvars["substrateLabelingDict"]
    assert isinstance(substrateLabelingDict, dict), 'substrateLabelingDict file did not contain a python dict named substrateLabelingDict as required!'

    # get list of existing output files we could resume from
    files = os.listdir() 
    matchingFiles = list(filter(lambda x: re.match('^' + outputBaseName + '(\d{3})_mcmc.npy$', x), files))
    fileNameIntegers = [int(re.sub('^' + outputBaseName + '(\d{3})_mcmc.npy', r'\1', filename)) for filename in matchingFiles]

    # define function to generate an output filename from a number
    def integerToFilename(outputBaseName, x, zfill=3):
        """Return the output file name for chain number x, such as base001_mcmc.npy."""
        return outputBaseName + str(x).zfill(zfill) + '_mcmc.npy'

    # get the filenames to resume from, and save to
    if len(fileNameIntegers) > 0:
        previousFileName = integerToFilename(outputBaseName, max(fileNameIntegers))
        newFileName = integerToFilename(outputBaseName, max(fileNameIntegers) + 1)
        print('resuming from previous instance ' + str(len(fileNameIntegers)))
    else:
        print('new run, not resuming')
        previousFileName = None
        newFileName = newFileName = integerToFilename(outputBaseName, 1)

    # Import and apply exchange fluxes from file
    model.readFluxConstraints(configData['fluxBoundsFile'])

    # set max flux
    limit = configData['maxFlux']
    for reaction in model.reactions:
        if reaction.lower_bound < -limit:
            reaction.lower_bound = -limit
        if reaction.upper_bound > limit:
            reaction.upper_bound = limit

    # run FBA to confirm model works
    model.optimize()

    # Read atom transitions and apply to model
    model.readAtomTransitions(configData['transitionsFile'])

    # Read mass distribution data
    importedMDVs = bayflux.readMassDistribution(model, configData['mdvFile'])

    # Pre-compile EMU network
    compiledData = bayflux.emuCompile(list(importedMDVs.data.keys()), model, substrateLabelingDict)

    # normpdf comes from bayflux.core.likelihood. This module used to define a
    # private copy, which had drifted from the shared one: it evaluated the pdf
    # outside the log domain and so returned -inf for residuals beyond 38.58
    # standard deviations, and it was not JIT compiled.

    # Add this before the logLikelihood function definition
    logLikelihood_values = []

    
    def logLikelihood(fluxVector, lcmsResults=importedMDVs, compiledData=compiledData, substrateLabelingDict=substrateLabelingDict, errorSD=configData['averageError']):
        """Log likelihood of a flux vector given the measured MDVs.

        The labeling is simulated and compared with each measured MDV under normal
        errors with standard deviation errorSD. A failed linear solve returns the
        lowest float32 value.
        """
        fluxVector = fluxVector + 1e-7 # make sure there's no zeros

        # simulate labeling distribution
        try:
            results = bayflux.simulateLabeling(compiledData, fluxVector, substrateLabelingDict)
        except np.linalg.LinAlgError:
            print('LinAlgError')
            return np.finfo(np.float32).min # return a really low probability if we get an error
        
        logLikelihood = 0
        
        # find overlap between results and predicted EMUs
        for emu, mdvs in lcmsResults.data.items():
            emuHash = hash(emu)
            found = False
            for singleSize in compiledData:
                if emuHash in singleSize['matrixCoords']['internalEMUs']['hashes']:
                    index = singleSize['matrixCoords']['internalEMUs']['hashes'].index(emuHash)
                    found = True
                    # zip together experimental results and simulated results into pairwise
                    # tuples
                    for mdv in mdvs:
                        for comparison in zip(mdv, results[singleSize['size']][index]):
                            logLikelihood += normpdf(comparison[0], comparison[1], errorSD)
            assert found == True, 'missing MDV in simulated results for ' + repr(emu)
        return logLikelihood
    
    
    # identify which task this is
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("task " + str(rank) + " of " + str(size))

    # compute a random seed unique to this task
    # and resume state, and use to create sampler
    thisSeed = (seed + (seed * rank)) ** (len(fileNameIntegers) + 1)
    #mcmcachr = cobra.sampling.MCMCACHRSampler(model, thinning=thinning, seed=thisSeed)
    mcmcachr = MCMCACHRSampler(model, thinning=thinning, seed=thisSeed)

    # compute a unique centerFile name to store the center for each
    # chain
    centerFile = outputBaseName + '_center' + str(rank)  + '.npy'

    # resume center if already saved
    if os.path.isfile(centerFile):
        # if center was already computed, load it from a file
        print(f'loading previously saved center from {centerFile}')
        mcmcachr.center = np.load(centerFile)
    else:
        # otherwise collect uniform samples to get the center
        mcmcachr.sample(warmupSamplesPerTask, fluxes=False, likelihood=False)
        np.save(centerFile, mcmcachr.center, allow_pickle=False)

    # if this isn't the first task, initialize the sampler
    # at the previous final position
    if previousFileName:
        previousData = np.load(previousFileName, mmap_mode='r')
        start = rank * samplesPertask
        end = start + samplesPertask
        mcmcachr.prev = np.array(previousData[end - 1])

    # now we sample considering probabilities, since we have a 'center'
    start_time = time.time()

    
    samples = np.array(mcmcachr.sample(samplesPertask, fluxes=False, likelihood=logLikelihood))
    print("--- %s seconds ---" % (time.time() - start_time))

    # if this isn't mpi rank 0, send the results to rank 0
    if rank  > 0:
        comm.Send(samples, dest=0)

    # if this is mpi rank 0, get results from the other sampler chains and save to file
    elif rank == 0:
        reactions = len(model.variables)
        data = np.zeros(shape=((samplesPertask * size), reactions), dtype=np.float64, order='C')
        data[0:samplesPertask] = samples
        for task in range(1, size):
            print("getting data from task " + str(task))
            start = task * samplesPertask
            end = start + samplesPertask 
            comm.Recv(data[start:end], source=task)
        print("result dimensions: " + str(data.shape))
        np.save(newFileName, data, allow_pickle=False)

    
