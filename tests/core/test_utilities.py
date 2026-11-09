"""Tests for bayflux.core.utilities.

Covers resource logging and both numerical precision helpers, including every
risk classification branch.
"""

import logging

import numpy as np
import pytest

from bayflux.core.utilities import (
    HIGH_RISK_LIKELIHOOD_RANGE,
    MODERATE_RISK_LIKELIHOOD_RANGE,
    OVERFLOW_THRESHOLD,
    analyzeNumericalPrecision,
    logSystemResources,
    quickPrecisionCheck,
)


def testLogSystemResourcesReportsMemoryAndCpu(caplog):
    with caplog.at_level(logging.INFO):
        logSystemResources()
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Memory usage:" in messages
    assert "CPU usage:" in messages


def testLogSystemResourcesReraisesUnexpectedErrors(monkeypatch, caplog):
    import bayflux.core.utilities as utilities

    def raiseOsError(pid):
        raise OSError("no such process table")

    monkeypatch.setattr(utilities.psutil, "Process", raiseOsError)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):
            logSystemResources()
    assert "Unexpected error" in " ".join(r.getMessage() for r in caplog.records)


def testLogSystemResourcesReraisesNoSuchProcess(monkeypatch):
    import bayflux.core.utilities as utilities

    def raiseNoSuchProcess(pid):
        raise utilities.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(utilities.psutil, "Process", raiseNoSuchProcess)
    with pytest.raises(utilities.psutil.NoSuchProcess):
        logSystemResources()


def testLogSystemResourcesReraisesAccessDenied(monkeypatch):
    import bayflux.core.utilities as utilities

    def raiseAccessDenied(pid):
        raise utilities.psutil.AccessDenied(pid)

    monkeypatch.setattr(utilities.psutil, "Process", raiseAccessDenied)
    with pytest.raises(utilities.psutil.AccessDenied):
        logSystemResources()


def testAnalyzeNumericalPrecisionCountsPairwiseDifferences():
    samples = np.zeros((3, 2))
    likelihoods = [-10.0, -11.0, -12.0]
    report = analyzeNumericalPrecision(samples, likelihoods, 2)

    # three states give six ordered pairs with i != j
    assert len(report["log_diffs"]) == 6
    assert report["safe_cases"] == 6
    assert report["overflow_cases"] == 0
    assert report["underflow_cases"] == 0
    assert report["risk_level"] == "LOW"


def testAnalyzeNumericalPrecisionFlagsHighRisk():
    samples = np.zeros((2, 2))
    likelihoods = [0.0, -2.0 * OVERFLOW_THRESHOLD]
    report = analyzeNumericalPrecision(samples, likelihoods, 1)

    assert report["overflow_cases"] == 1
    assert report["underflow_cases"] == 1
    assert report["safe_cases"] == 0
    assert report["risk_level"] == "HIGH"


def testAnalyzeNumericalPrecisionFlagsModerateRisk():
    # One extreme state among many safe ones keeps the problematic fraction
    # below the one percent threshold that separates MODERATE from HIGH. A
    # single extreme state contributes 2/numStates of all ordered pairs, so
    # MODERATE needs more than 200 states, which is well beyond any realistic
    # numProposals. This branch is therefore effectively unreachable in
    # production and is covered here only for completeness.
    numStates = 250
    likelihoods = np.linspace(0.0, 1.0, numStates)
    likelihoods[-1] = -2.0 * OVERFLOW_THRESHOLD
    samples = np.zeros((numStates, 2))
    report = analyzeNumericalPrecision(samples, likelihoods, numStates - 1)

    assert report["overflow_cases"] + report["underflow_cases"] > 0
    assert report["risk_level"] == "MODERATE"


def testAnalyzeNumericalPrecisionAcceptsNumpyInput():
    report = analyzeNumericalPrecision(
        np.zeros((2, 3)), np.array([-1.0, -2.0]), 1
    )
    assert isinstance(report["log_diffs"], np.ndarray)
    assert report["risk_level"] == "LOW"


@pytest.mark.parametrize(
    "likelihoods, expected",
    [
        ([0.0, -1.0], "LOW"),
        ([0.0, -MODERATE_RISK_LIKELIHOOD_RANGE], "LOW"),
        ([0.0, -MODERATE_RISK_LIKELIHOOD_RANGE - 1.0], "MODERATE"),
        ([0.0, -HIGH_RISK_LIKELIHOOD_RANGE], "MODERATE"),
        ([0.0, -HIGH_RISK_LIKELIHOOD_RANGE - 1.0], "HIGH"),
    ],
)
def testQuickPrecisionCheckClassifiesRange(likelihoods, expected):
    assert quickPrecisionCheck(likelihoods) == expected


def testQuickPrecisionCheckAcceptsNumpyInput():
    assert quickPrecisionCheck(np.array([-5.0, -5.0, -5.0])) == "LOW"
