# -*- coding: utf-8 -*-

"""Provide MCMC ACHR sampler."""

from __future__ import absolute_import, division

import numpy as np
import pandas

from cobra.sampling.hr_sampler import HRSampler, step

import sys


class MCMCACHRSampler(HRSampler):
    """MCMC Artificial Centering Hit-and-Run (ACHR) sampler.

    An ACHR sampler [1]_ extended to take a prior and a likelihood function, so
    that samples are accepted or rejected with the Metropolis test [2]_. New
    directions come from the center and the initial warmup points only.

    Run the sampler first without a likelihood or prior, for enough samples that
    the center converges. Then run it again with them. While a likelihood or prior
    is given the center is locked, which keeps the chain Markovian.

    Memory use grows roughly as (2 * number of reactions)^2, so large models can
    need a few GB of RAM.

    Parameters
    ----------
    model : cobra.Model
        The model to sample from.
    thinning : int, optional
        Return a sample every this many steps.
    nproj : int > 0, optional
        How often to reproject the sample into the feasible space. Lower it if
        sampler.validate reports many equality constraint violations.
    seed : int > 0, optional
        Random seed. Defaults to the current time stamp.

    Attributes
    ----------
    prev : numpy.array
        The latest flux sample.
    center : numpy.array
        Mean of all previous samples.
    n_samples : int
        Number of samples generated so far.
    bestSample : tuple
        (posterior, sample) for the highest posterior seen so far, or None.

    Other attributes (model, problem, warmup, retries, fwd_idx, rev_idx) are
    inherited from the cobrapy ACHR sampler.

    References
    ----------
    .. [1] Kaufman DE, Smith RL. Direction Choice for Accelerated Convergence in
       Hit-and-Run Sampling. Operations Research 46(1):84-95 (1998).
       https://doi.org/10.1287/opre.46.1.84
    .. [2] Metropolis N et al. Equation of State Calculations by Fast Computing
       Machines. J. Chem. Phys. 21(6):1087 (1953).
       https://doi.org/10.1063/1.1699114
    """

    def __init__(self, model, thinning=100, nproj=None, seed=None):
        """Initialize a new MCMCACHRSampler."""

        super(MCMCACHRSampler, self).__init__(model, thinning, nproj=nproj, seed=seed)
        self.generate_fva_warmup(includeReversible=True)
        self.prev = self.center = self.warmup.mean(axis=0)
        np.random.seed(self._seed)

        # create a variable to store the best point we sampled
        self.bestSample = None

    def __single_iteration(self, lockCenter=False, validatecheck=False):
        """Take one hit-and-run step from the previous sample.

        If lockCenter is set, the center is not updated.
        """

        nmax = 10 #tries to find new valid sample maximum 10 times for each situation

        pi = np.random.randint(self.n_warmup)

        # mix in the original warmup points to not get stuck
        delta = self.warmup[pi, ] - self.center
        ## create testprev to check if current sample is valid
        testprev = step(self, self.prev, delta)

        ###########################
        #optional validation for posterior samples:
        if validatecheck:
            counter = 0
            while counter<=nmax and not any(element in 'v' for element in self.validate(np.transpose(testprev), feas_tol=1e-6, bounds_tol=1e-6)):#first sample: #input have to be netsamples and in form samples x reactions#, feas_tol=1e-6, bounds_tol=1e-6)
                if counter==nmax:
                    print('Tried to find valid sample', nmax, 'times without success')
                    sys.exit()
                print('searching new valid sample as validate output=', self.validate(np.transpose(testprev)))#, feas_tol=1e-6, bounds_tol=1e-6))
                pi = np.random.randint(self.n_warmup)
                delta = self.warmup[pi, ] - self.center
                testprev = step(self, self.prev, delta)
                counter += 1
        self.prev = testprev
        ###########################

        if self.problem.homogeneous and (self.n_samples *
                                         self.thinning % self.nproj == 0):
            self.prev = self._reproject(self.prev)
            if not lockCenter:
                self.center = self._reproject(self.center)
        if not lockCenter:
            self.center = ((self.n_samples * self.center) / (self.n_samples + 1) +
                           self.prev / (self.n_samples + 1))
        self.n_samples += 1

    def sample(self, n, fluxes=True, likelihood=None, prior=None, validatecheck=False):
        """Generate a set of samples.

        This is the basic sampling function for all hit-and-run samplers,
        extended to support prior and likelihood functions for MCMC
        Bayesian inference of metabolic fluxes.

        Parameters
        ----------
        n : int
            The number of samples that are generated at once.
        fluxes : boolean
            Whether to return fluxes or the internal solver variables. If set
            to False will return a variable for each forward and backward flux
            as well as all additional variables you might have defined in the
            model.
        likelihood : function
            A python function which will take as an argument the flux vector
            (current sample) and return a log likelihood, computed by comparing
            the current flux vector to experimental data. This will be used to
            accept or reject samples per the Metropolis algorithm. If this value
            is passed (e.g. not False), the ACHR center will be locked,
            and not be updated. If the default (None) is used for both
            the prior and likelihood options, normal ACHR sampling will take
            place, with updates to the center on each sample.
        prior : function
            A python function which will take as an argument the flux vector
            (current sample) and return a log prior. This will be used to
            accept or reject samples per the Metropolis algorithm. If this value
            is passed (e.g. not None), the ACHR center will be locked,
            and not be updated. If the default (None) is used for both
            the prior and likelihood options, normal ACHR sampling will take
            place, with updates to the center on each sample.
        validatecheck : boolean
            Checking if current sample is valid or not. If it is True goes back
            to previous sample and finds a new valid sample (repeated for up to nmax=10 times).

        Returns
        -------
        numpy.matrix
            Returns a matrix with `n` rows, each containing a flux sample.

        Notes
        -----
        Performance of this function linearly depends on the number
        of reactions in your model and the thinning factor.

        """
        samples = np.zeros((n, self.warmup.shape[1]))

        # store the log posterior of the previous sample
        previousPosterior = False#None
        savePrev = None#np.empty(len(self.prev))# False#None
        totalSamples = 0
        rejections = 0

        # determine if we are doing centering samples, or MCMC samples
        # with a locked center
        if prior or likelihood:
            lockCenter = True
        else:
            lockCenter = False

        for i in range(1, self.thinning * n + 1):

            self.__single_iteration(lockCenter=lockCenter, validatecheck=validatecheck)

            totalSamples += 1
            if lockCenter:
                if likelihood:
                    newLikelihood = likelihood(self.prev)
                else:
                    newLikelihood = 0
                if prior:
                    newPrior = prior(self.prev)
                else:
                    newPrior = 0
                newPosterior = newLikelihood + newPrior
                acceptProbability = newPosterior - previousPosterior
                #print(f"--> Iteration {i}: logAcceptProbability (new): {acceptProbability:.6f} | acceptProbability (new): {np.exp(acceptProbability):.6f}")
                if not previousPosterior:
                    # always accept on first iteration
                    previousPosterior = newPosterior
                    savePrev = self.prev
                elif np.log(np.random.rand()) < acceptProbability:
                    #print(f"--> acceptProbability (old): {1-acceptProbability:.6f}, acceptProbability (new): {acceptProbability:.6f}")
                    # then accept if probability is high enough
                    previousPosterior = newPosterior
                    savePrev = self.prev
                    if not self.bestSample:
                        self.bestSample = (newPosterior, self.prev)
                    else:
                        if newPosterior > self.bestSample[0]:
                            self.bestSample = (newPosterior, self.prev)
                else:
                    # reject (keep previous state)
                    self.prev = savePrev
                    rejections += 1

            if i % self.thinning == 0:
                samples[i // self.thinning - 1, :] = self.prev

        print('acceptance rate: ' + str(float(totalSamples - rejections) / float(totalSamples)))

        if fluxes:
            names = [r.id for r in self.model.reactions]

            return pandas.DataFrame(
                samples[:, self.fwd_idx] - samples[:, self.rev_idx], columns=names,
            )
        else:
            names = [v.name for v in self.model.variables]

            return pandas.DataFrame(samples, columns=names)





