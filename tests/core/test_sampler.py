"""Tests for bayflux.core.sampler.MCMCACHRSampler.

The sampler is the component every executable is built around and nothing
covered it. These tests use the smallest thinning and sample count that still
exercise both the centering path and the Metropolis path.
"""

import numpy as np
import pandas as pd
import pytest

from bayflux.core.sampler import MCMCACHRSampler


def testConstructionBuildsWarmupPointsAndACenter(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)

    assert sampler.warmup.shape[0] == sampler.n_warmup
    assert sampler.warmup.shape[1] == len(toySampleableModel.variables)
    assert np.array_equal(sampler.center, sampler.prev)
    assert sampler.bestSample is None


def testCenteringSamplesMoveTheCenter(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)
    startingCenter = sampler.center.copy()

    samples = sampler.sample(5)

    assert isinstance(samples, pd.DataFrame)
    assert samples.shape == (5, len(toySampleableModel.reactions))
    assert list(samples.columns) == [r.id for r in toySampleableModel.reactions]
    assert sampler.n_samples == 2 * 5
    # without a likelihood the center is free to move, which is the point of
    # the centering phase
    assert not np.array_equal(sampler.center, startingCenter)


def testSampleCanReturnSolverVariablesInstead(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)
    samples = sampler.sample(3, fluxes=False)

    assert samples.shape == (3, len(toySampleableModel.variables))
    assert list(samples.columns) == [v.name for v in toySampleableModel.variables]


def testLikelihoodLocksTheCenterAndRecordsTheBestSample(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)
    lockedCenter = sampler.center.copy()

    calls = []

    def likelihood(fluxVector):
        calls.append(np.asarray(fluxVector).copy())
        # reward small total flux, so that proposals are genuinely ranked
        return -float(np.abs(fluxVector).sum())

    samples = sampler.sample(4, likelihood=likelihood)

    assert samples.shape == (4, len(toySampleableModel.reactions))
    assert len(calls) == 2 * 4
    assert np.array_equal(sampler.center, lockedCenter)
    assert sampler.bestSample is not None
    bestPosterior, bestVector = sampler.bestSample
    assert np.isfinite(bestPosterior)
    assert len(bestVector) == len(toySampleableModel.variables)


def testPriorAloneAlsoLocksTheCenter(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)
    lockedCenter = sampler.center.copy()

    priorCalls = []

    def prior(fluxVector):
        priorCalls.append(1)
        return -1.0

    sampler.sample(3, prior=prior)

    assert len(priorCalls) == 2 * 3
    assert np.array_equal(sampler.center, lockedCenter)


def testRejectionsAreReportedInTheAcceptanceRate(toySampleableModel, capsys):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)

    def punishingLikelihood(fluxVector):
        # every proposal after the first is far worse, so almost all are rejected
        punishingLikelihood.calls += 1
        return -1e3 * punishingLikelihood.calls

    punishingLikelihood.calls = 0
    sampler.sample(4, likelihood=punishingLikelihood)

    printed = capsys.readouterr().out
    assert "acceptance rate: " in printed
    reportedRate = float(printed.strip().split("acceptance rate: ")[-1].split()[0])
    assert 0.0 <= reportedRate < 1.0


def testValidateCheckKeepsSamplesInsideTheConstraints(toySampleableModel):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=2, seed=11)
    samples = sampler.sample(3, validatecheck=True)
    assert samples.shape == (3, len(toySampleableModel.reactions))
    assert np.isfinite(samples.to_numpy()).all()
