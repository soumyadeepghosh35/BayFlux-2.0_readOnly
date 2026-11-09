"""End to end test of bayflux.core.findCenter, the standalone centering utility.

This module used to be unimportable: it applied bare @profile decorators, the
name line_profiler's kernprof injects into builtins, and nothing in the module
defined or imported it, so every invocation failed immediately with NameError.
Those decorators are now commented out and the module imports like any other,
so these tests import it directly.

Both invocations work: `python -m bayflux.core.findCenter config.yaml` and
`python bayflux/core/findCenter.py config.yaml`. The second only works because
findCenter.py drops its own directory from sys.path before importing bayflux,
so the tests below cover that guard from both directions.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def findCenterModule():
    """Import findCenter fresh for each test.

    main() writes files named from the configuration, and the tests below run
    it more than once, so a clean import per test keeps module level state from
    leaking between them.
    """
    sys.modules.pop("bayflux.core.findCenter", None)
    try:
        yield importlib.import_module("bayflux.core.findCenter")
    finally:
        sys.modules.pop("bayflux.core.findCenter", None)


def testFindCenterIsImportable():
    # A bare @profile decorator anywhere at module level breaks this, and with
    # it every way of running the utility except `kernprof -l`
    sys.modules.pop("bayflux.core.findCenter", None)
    module = importlib.import_module("bayflux.core.findCenter")
    assert callable(module.autoCenter)
    assert callable(module.main)


def runInFreshInterpreter(statement):
    """Run one statement in a new interpreter, outside the pytest process.

    conftest.py imports bayflux before anything else, which applies the numpy
    compatibility shim for the whole session. That hides import order problems
    from every in process test, so this check has to leave the process.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True, text=True, env=environment, cwd=str(REPOSITORY_ROOT),
    )


def runScript(scriptPath, *arguments):
    """Run a file by path, the way a person would from a shell.

    Python puts the script's own directory at the front of sys.path, ahead of
    anything in PYTHONPATH. That is what makes running a file inside
    bayflux/core different from importing it.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, str(scriptPath), *arguments],
        capture_output=True, text=True, env=environment, cwd=str(REPOSITORY_ROOT),
    )


def runModule(moduleName, *arguments):
    """Run a module with -m, which is the supported way to invoke a utility.

    -m imports the package before the module, so bayflux/__init__.py executes
    and applies the numpy compatibility shim. This is the same resolution the
    console scripts in setup.py use.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", moduleName, *arguments],
        capture_output=True, text=True, env=environment, cwd=str(REPOSITORY_ROOT),
    )


def testRunningFindCenterThroughThePackageWorks():
    # -m executes bayflux/__init__.py first, so the numpy shim is in place
    # before cobra is imported
    result = runInFreshInterpreter("import bayflux.core.findCenter")
    assert result.returncode == 0, result.stderr


def testRunningFindCenterAsAModuleReachesArgparse():
    """The supported invocation, exercised end to end through a subprocess.

    Invoked with no arguments, a working module gets as far as argparse and
    stops there, so argparse's own usage message on stderr is proof that every
    import along the way succeeded. Anything else in the output is an import
    failure.
    """
    result = runModule("bayflux.core.findCenter")
    assert result.returncode == 2, result.stderr
    assert "config_file" in result.stderr, result.stderr


def testRunningFindCenterAsAScriptReachesArgparse():
    """Running the file by path works, which takes a guard to achieve.

    Python puts a script's own directory at the front of sys.path, ahead of
    PYTHONPATH, so bayflux/core/bayflux.py would otherwise satisfy
    `import bayflux` before the package does and bayflux/__init__.py, with the
    numpy compatibility shim the pinned COBRApy fork needs, would never run.
    findCenter.py drops that entry before importing bayflux, so the package
    wins the name.

    Invoked with no arguments a working script gets as far as argparse and
    stops there, so its usage message on stderr is proof that every import
    along the way succeeded. Anything else is an import failure.
    """
    scriptPath = REPOSITORY_ROOT / "bayflux" / "core" / "findCenter.py"
    result = runScript(scriptPath)
    assert result.returncode == 2, result.stderr
    assert "config_file" in result.stderr, result.stderr


def testImportingFindCenterLeavesTheCallersSysPathIntact():
    """The guard must fire only when the module is run as a script.

    findCenter.py pops sys.path[0] when it equals its own directory. Under any
    ordinary import that entry belongs to the caller, and removing it would
    break the importing program. Only the first entry is checked here, because
    dependencies are free to append to sys.path during their own import.
    """
    result = runInFreshInterpreter(
        "import sys; first = sys.path[0]; "
        "import bayflux.core.findCenter; "
        "print('kept' if sys.path and sys.path[0] == first else 'lost')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("kept"), result.stdout


def testImportingBayfluxFromInsideCoreResolvesToTheModule():
    """Pin the shadowing the guard above defends against.

    find_spec locates the module without executing it, which matters because
    importing it is exactly what fails. This is the precise statement of the
    collision: if bayflux/core/bayflux.py is ever renamed, this test and the
    guard in findCenter.py are retired together.
    """
    coreDirectory = REPOSITORY_ROOT / "bayflux" / "core"
    result = runInFreshInterpreter(
        "import sys, importlib.util; sys.path.insert(0, %r); "
        "print(importlib.util.find_spec('bayflux').origin)" % str(coreDirectory)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(os.path.join("core", "bayflux.py")), (
        result.stdout
    )


def testAutoCenterCollectsOneCenterPerStep(findCenterModule, toySampleableModel):
    from bayflux.core.sampler import MCMCACHRSampler

    sampler = MCMCACHRSampler(toySampleableModel, thinning=1, seed=7)
    centers = findCenterModule.autoCenter(
        sampler=sampler, samples=6, centerPoints=3, confirmValidity=False
    )

    assert centers.shape == (3, len(toySampleableModel.variables))
    assert np.isfinite(centers).all()
    # the center moves as more samples arrive, which is the drift being measured
    assert not np.array_equal(centers[0], centers[-1])


def testAutoCenterCanConfirmSampleValidity(findCenterModule, toySampleableModel):
    from bayflux.core.sampler import MCMCACHRSampler

    sampler = MCMCACHRSampler(toySampleableModel, thinning=1, seed=7)
    centers = findCenterModule.autoCenter(
        sampler=sampler, samples=4, centerPoints=2, confirmValidity=True
    )

    assert centers.shape == (2, len(toySampleableModel.variables))
    assert np.isfinite(centers).all()


def testMainWritesCenterAndDriftFiles(
    findCenterModule, toyRunDirectory, monkeypatch, capsys
):
    runDirectory = toyRunDirectory(
        outputBaseName="outputdata/toy", centeringSamplesPerTask=3, thinning=1
    )
    monkeypatch.setattr(sys, "argv", ["findCenter", "config.yaml"])

    findCenterModule.main()

    printed = capsys.readouterr().out
    assert "Metabolic model information" in printed
    assert "Calculating drift centers" in printed
    assert "Calculating center by collecting uniform samples" in printed

    outputDirectory = runDirectory / "outputdata"
    center = np.load(outputDirectory / "toy_center.npy")
    drift = np.load(outputDirectory / "toy_centerDrift.npy")
    data = np.load(outputDirectory / "toy_centerData.npy")

    assert center.shape == (len(np.load(outputDirectory / "toy_center.npy")),)
    assert np.isfinite(center).all()
    assert drift.shape[0] == 3
    assert data.shape[0] >= 1


def testMainReusesAnExistingCenter(
    findCenterModule, toyRunDirectory, monkeypatch, capsys
):
    toyRunDirectory(
        outputBaseName="outputdata/toy", centeringSamplesPerTask=3, thinning=1
    )
    monkeypatch.setattr(sys, "argv", ["findCenter", "config.yaml"])

    findCenterModule.main()
    capsys.readouterr()

    findCenterModule.main()

    assert "Loading previously saved center" in capsys.readouterr().out


def testMalformedConfigExitsWithAnError(
    findCenterModule, toyRunDirectory, monkeypatch, capsys
):
    runDirectory = toyRunDirectory(outputBaseName="outputdata/toy")
    (runDirectory / "config.yaml").write_text("thinning: [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["findCenter", "config.yaml"])

    with pytest.raises(SystemExit):
        findCenterModule.main()

    assert "Error loading configuration file" in capsys.readouterr().out
