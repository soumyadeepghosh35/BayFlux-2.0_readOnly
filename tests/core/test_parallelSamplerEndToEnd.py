"""End to end test of bayflux.core.bayflux, the parallel bayflux executable.

The parallel sampler coordinates worker processes through shared memory, so it
can only be exercised as a whole. The sample counts here are the smallest that
still drive a full MP-MCMC iteration for every proposal.
"""

import sys

import numpy as np
import pytest


def testParallelSamplerProducesPosteriorSamples(toyRunDirectory, monkeypatch, capsys):
    runDirectory = toyRunDirectory(
        outputBaseName="outputdata/par",
        bayesianSamplesPerTask=6,
        numProposals=2,
        subsamplesPerIteration=3,
    )
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])

    parallelSampler = pytest.importorskip("bayflux.core.bayflux")
    parallelSampler.main()

    assert "Job successfully completed" in capsys.readouterr().out

    samples = np.load(runDirectory / "outputdata" / "par001.npy")
    assert samples.ndim == 2
    assert samples.shape[0] == 6
    assert np.isfinite(samples).all()


def testParallelSamplerRunsWithMoreProposalsThanSubsamples(
    toyRunDirectory, monkeypatch, capsys
):
    runDirectory = toyRunDirectory(
        outputBaseName="outputdata/par",
        bayesianSamplesPerTask=4,
        numProposals=3,
        subsamplesPerIteration=2,
    )
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])

    parallelSampler = pytest.importorskip("bayflux.core.bayflux")
    parallelSampler.main()

    assert "Job successfully completed" in capsys.readouterr().out
    samples = np.load(runDirectory / "outputdata" / "par001.npy")
    assert samples.shape[0] == 4


def testSecondRunReusesTheSavedCenter(toyRunDirectory, monkeypatch, capsys):
    toyRunDirectory(outputBaseName="outputdata/par", bayesianSamplesPerTask=4)
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])
    parallelSampler = pytest.importorskip("bayflux.core.bayflux")

    parallelSampler.main()
    capsys.readouterr()

    # the output directory already exists on the second run, and so does the
    # saved center, so both resume paths are taken
    parallelSampler.main()

    assert "Loading previously saved center" in capsys.readouterr().out


def testMalformedConfigExitsWithAnError(toyRunDirectory, monkeypatch, capsys):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/par")
    (runDirectory / "config.yaml").write_text("thinning: [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])
    parallelSampler = pytest.importorskip("bayflux.core.bayflux")

    with pytest.raises(SystemExit) as exitInfo:
        parallelSampler.main()

    assert exitInfo.value.code == 1
    assert "Error parsing YAML configuration" in capsys.readouterr().out


def testUnreadableModelExitsWithAnError(toyRunDirectory, monkeypatch, capsys):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/par")
    (runDirectory / "fig3toyModel.xml").write_text("this is not SBML\n")
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])
    parallelSampler = pytest.importorskip("bayflux.core.bayflux")

    with pytest.raises(SystemExit) as exitInfo:
        parallelSampler.main()

    assert exitInfo.value.code == 1
    assert "Error reading SBML model" in capsys.readouterr().out


def testBrokenSubstrateLabelingFileExitsWithAnError(
    toyRunDirectory, monkeypatch, capsys
):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/par")
    (runDirectory / "substrateLabelingDict.py").write_text(
        "raise RuntimeError('deliberate failure')\n"
    )
    monkeypatch.setattr(sys, "argv", ["bayflux", "config.yaml"])
    parallelSampler = pytest.importorskip("bayflux.core.bayflux")

    with pytest.raises(SystemExit) as exitInfo:
        parallelSampler.main()

    assert exitInfo.value.code == 1
    assert "Error executing substrate labeling file" in capsys.readouterr().out
