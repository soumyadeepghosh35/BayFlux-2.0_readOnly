"""Tests for bayflux.core.fluxes.FluxVector.

Covers construction from both a dict of forward and reverse fluxes and a raw
array, the validation assertions on that dict, and the display helpers.
"""

import numpy as np
import pandas as pd
import pytest

import bayflux


def testConstructionFromDictOrdersVariablesCorrectly(toyModel):
    fluxes = bayflux.FluxVector(toyModel, {"a_b": [100.0, 0.0]})
    variableIndex = {variable: index
                     for index, variable in enumerate(toyModel.variables)}
    reaction = toyModel.reactions.get_by_id("a_b")

    assert len(fluxes.fluxes) == len(toyModel.variables)
    assert fluxes[variableIndex[reaction.forward_variable]] == 100.0
    assert fluxes[variableIndex[reaction.reverse_variable]] == 0.0
    assert fluxes.fluxes.sum() == 100.0


def testConstructionFromArrayIsStoredVerbatim(toyModel):
    rawFluxes = np.arange(len(toyModel.variables), dtype=np.float64)
    fluxes = bayflux.FluxVector(toyModel, rawFluxes)
    assert np.array_equal(fluxes.fluxes, rawFluxes)
    assert fluxes.model is toyModel


def testReversibleReactionAcceptsReverseFlux(toyModel):
    fluxes = bayflux.FluxVector(toyModel, {"b_d": [110.0, 50.0]})
    assert fluxes.fluxes.sum() == 160.0


def testUnknownReactionIsRejected(toyModel):
    with pytest.raises(AssertionError, match="Reaction name not in model"):
        bayflux.FluxVector(toyModel, {"not_a_reaction": [1.0, 0.0]})


def testFluxPairMustHaveLengthTwo(toyModel):
    with pytest.raises(AssertionError, match="length 2 flux"):
        bayflux.FluxVector(toyModel, {"a_b": [1.0, 0.0, 3.0]})


def testNegativeForwardFluxIsRejected(toyModel):
    with pytest.raises(AssertionError, match="Invalid negative flux"):
        bayflux.FluxVector(toyModel, {"a_b": [-1.0, 0.0]})


def testNegativeReverseFluxIsRejected(toyModel):
    with pytest.raises(AssertionError, match="Invalid negative flux"):
        bayflux.FluxVector(toyModel, {"a_b": [1.0, -1.0]})


def testIrreversibleReactionRejectsReverseFlux(toyModel):
    with pytest.raises(AssertionError, match="non-reversible reaction"):
        bayflux.FluxVector(toyModel, {"a_b": [100.0, 10.0]})


def testWrongLengthArrayIsRejected(toyModel):
    with pytest.raises(AssertionError, match="Number of variables in model"):
        bayflux.FluxVector(toyModel, np.zeros(3))


def testToSeriesIndexesByVariableName(toyFluxes, toyModel):
    series = toyFluxes.to_Series()
    assert isinstance(series, pd.Series)
    assert len(series) == len(toyModel.variables)
    assert list(series.index) == [v.name for v in toyModel.variables]
    assert series.sum() == pytest.approx(sum(sum(pair)
                                             for pair in ({"a_b": [100, 0],
                                                           "b_d": [110, 50],
                                                           "b_ec": [20, 0],
                                                           "bc_de": [20, 0],
                                                           "d_f": [80, 0]}).values()))


def testGetItemReadsASingleVariable(toyFluxes):
    assert toyFluxes[0] == toyFluxes.fluxes[0]


def testReprMatchesTheSeriesRepr(toyFluxes):
    assert repr(toyFluxes) == repr(toyFluxes.to_Series())


def testReprHtmlProducesATable(toyFluxes):
    html = toyFluxes._repr_html_()
    assert "<table" in html
    assert "Flux" in html
