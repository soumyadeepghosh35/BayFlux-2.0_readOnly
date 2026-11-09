"""End to end test of bayflux.core.commandLine, the bayflux_serial executable.

Nothing covered this module, which is how a private duplicate of normpdf came
to live in it unnoticed. These tests run main() the way the console script
does, against the staged toy input set, with sample counts small enough to
finish in seconds.
"""

import sys

import numpy as np
import pytest
import yaml


def testSerialSamplerProducesPosteriorSamples(toyRunDirectory, monkeypatch, capsys):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/toy")
    monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])

    from bayflux.core.commandLine import main
    main()

    printed = capsys.readouterr().out
    assert "new run, not resuming" in printed
    assert "task 0 of 1" in printed
    assert "result dimensions:" in printed

    outputFile = runDirectory / "outputdata" / "toy001_mcmc.npy"
    centerFile = runDirectory / "outputdata" / "toy_center0.npy"
    assert outputFile.is_file()
    assert centerFile.is_file()

    samples = np.load(outputFile)
    assert samples.shape[0] == 5
    assert samples.dtype == np.float64
    assert np.isfinite(samples).all()
    # sampling is done with fluxes=False, so the columns are solver variables:
    # a forward and a reverse variable for each of the eight reactions
    assert samples.shape[1] == 16

    center = np.load(centerFile)
    assert center.shape == (16,)
    assert np.isfinite(center).all()


def testSecondRunReusesTheSavedCenter(toyRunDirectory, monkeypatch, capsys):
    toyRunDirectory(outputBaseName="outputdata/toy")
    monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])

    from bayflux.core.commandLine import main
    main()
    capsys.readouterr()

    main()

    printed = capsys.readouterr().out
    assert "loading previously saved center" in printed


def testMaxFluxClampsTheReactionBounds(toyRunDirectory, monkeypatch):
    # maxFlux is applied after the constraints file, so a bound wider than the
    # limit must come back clamped
    toyRunDirectory(outputBaseName="outputdata/toy", maxFlux=50.0)
    monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])

    import bayflux.core.commandLine as commandLine
    originalSampler = commandLine.MCMCACHRSampler
    captured = {}

    def recordingSampler(model, *args, **kwargs):
        captured["bounds"] = {
            reaction.id: (reaction.lower_bound, reaction.upper_bound)
            for reaction in model.reactions
        }
        return originalSampler(model, *args, **kwargs)

    monkeypatch.setattr(commandLine, "MCMCACHRSampler", recordingSampler)
    commandLine.main()

    assert captured["bounds"]["a_b"] == (0.0, 50.0)
    assert captured["bounds"]["b_d"] == (-50.0, 50.0)


def testMisnamedSubstrateLabelingDictIsRejected(toyRunDirectory, monkeypatch):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/toy")
    (runDirectory / "substrateLabelingDict.py").write_text("notTheRightName = {}\n")
    monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])

    from bayflux.core.commandLine import main
    with pytest.raises(KeyError):
        main()


def testMalformedConfigIsReported(toyRunDirectory, monkeypatch, capsys):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/toy")
    (runDirectory / "config.yaml").write_text("thinning: [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])

    from bayflux.core.commandLine import main
    # the YAML error is caught and printed, then the missing keys surface
    with pytest.raises(UnboundLocalError):
        main()
    assert "while parsing" in capsys.readouterr().out
