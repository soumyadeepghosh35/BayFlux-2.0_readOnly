"""Tests for bayflux.core.likelihood.logLikelihood and the normpdf coercion.

logLikelihood is the function the samplers call once per proposal, and nothing
covered it. These tests run it against the Fig. 3 toy network, where the
expected labeling of F is known.
"""

import numpy as np
import pytest

import bayflux
from bayflux.core.likelihood import (
    MIN_LOG_LIKELIHOOD,
    asFloatScalar,
    logLikelihood,
    normpdf,
)


def testLogLikelihoodRewardsAgreementWithTheSimulation(
    toyModel, toyFluxes, toyCompiledData, toySubstrateLabeling, toyMassDistribution
):
    matched = logLikelihood(
        toyFluxes.fluxes,
        toyMassDistribution,
        toyCompiledData,
        toySubstrateLabeling,
        errorSD=0.0076875,
    )
    assert np.isfinite(matched)

    # a deliberately wrong MDV for the same EMU must score lower
    emu = next(iter(toyMassDistribution.data))
    toyMassDistribution.data[emu] = [[0.25, 0.25, 0.25, 0.25]]
    mismatched = logLikelihood(
        toyFluxes.fluxes,
        toyMassDistribution,
        toyCompiledData,
        toySubstrateLabeling,
        errorSD=0.0076875,
    )
    assert mismatched < matched


def testLogLikelihoodSumsOverReplicates(
    toyModel, toyFluxes, toyCompiledData, toySubstrateLabeling, toyMassDistribution
):
    emu = next(iter(toyMassDistribution.data))
    single = logLikelihood(
        toyFluxes.fluxes, toyMassDistribution, toyCompiledData,
        toySubstrateLabeling, errorSD=0.0076875,
    )

    toyMassDistribution.data[emu].append(list(toyMassDistribution.data[emu][0]))
    doubled = logLikelihood(
        toyFluxes.fluxes, toyMassDistribution, toyCompiledData,
        toySubstrateLabeling, errorSD=0.0076875,
    )
    assert doubled == pytest.approx(2.0 * single)


def testLogLikelihoodRejectsAnEmuThatWasNotCompiled(
    toyModel, toyFluxes, toyCompiledData, toySubstrateLabeling
):
    strayEmu = bayflux.EMU(toyModel.metabolites.get_by_id("C"), [1, 2])
    massDistribution = bayflux.MassDistribution(toyModel)
    massDistribution.data[strayEmu].append([0.4, 0.3, 0.3])

    with pytest.raises(AssertionError, match="Missing experimental EMU"):
        logLikelihood(
            toyFluxes.fluxes, massDistribution, toyCompiledData,
            toySubstrateLabeling, errorSD=0.0076875,
        )


def testLogLikelihoodReturnsTheFloorOnLinearAlgebraFailure(
    monkeypatch, toyFluxes, toyCompiledData, toySubstrateLabeling,
    toyMassDistribution,
):
    def raiseLinAlgError(compiledData, fluxVector, substrateLabelingDict):
        raise np.linalg.LinAlgError("singular EMU matrix")

    monkeypatch.setattr(bayflux, "simulateLabeling", raiseLinAlgError)
    result = logLikelihood(
        toyFluxes.fluxes, toyMassDistribution, toyCompiledData,
        toySubstrateLabeling, errorSD=0.0076875,
    )
    assert result == MIN_LOG_LIKELIHOOD


def testAsFloatScalarAcceptsScalarsAndZeroDimensionalArrays():
    assert asFloatScalar(3) == 3.0
    assert asFloatScalar(3.5) == 3.5
    assert asFloatScalar(np.float64(2.5)) == 2.5
    assert asFloatScalar(np.array(1.25)) == 1.25


def testNormpdfAcceptsEveryScalarLikeInput():
    expected = normpdf(1.0, 0.0, 1.0)
    assert normpdf(1, 0, 1) == pytest.approx(expected)
    assert normpdf(np.float64(1.0), np.float64(0.0), np.float64(1.0)) == \
        pytest.approx(expected)
    assert normpdf(np.array(1.0), np.array(0.0), np.array(1.0)) == \
        pytest.approx(expected)


def testNormpdfRejectsHigherDimensionalArrays():
    # a shaped array is a caller mistake, not a scalar measurement
    with pytest.raises(Exception):
        normpdf(np.array([1.0, 2.0]), 0.0, 1.0)


def testNormpdfIsFiniteFarIntoTheTail():
    # the previous non-log-domain form returned -inf beyond 38.58 sigma
    assert np.isfinite(normpdf(1000.0, 0.0, 1.0))
    assert normpdf(1000.0, 0.0, 1.0) == pytest.approx(-500000.9189385332)
