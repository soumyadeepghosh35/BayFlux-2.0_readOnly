# BayFlux 2.0 test suite and code coverage

This directory holds the automated tests for BayFlux 2.0. This document covers
how to run them, how to measure coverage correctly, what the current coverage
is, and what is deliberately left uncovered.

## Contents

    tests/
      README.md                    this file
      conftest.py                  shared fixtures and collection settings
      core/                        22 test modules
      fixtures/toyModel/           the input set the end to end tests stage
      fixtures/input_data/         E. coli genome scale inputs, held for a
                                   future genome scale test

Everything under `tests/` runs from a fresh clone with no path editing and no
data that is not already in the repository. 

## Running the tests

Run everything from the repository root, so that pytest's rootdir resolves
there and picks up `setup.cfg`:

    python3 -m pytest tests -q

The whole suite takes about ~30 seconds. Two things must be true before
pytest can import anything:

- `PYTHONPATH` must reach the repository root. Inside the released image this
  is already set (`ENV PYTHONPATH=/bayflux`).
- An `emusFortran` extension module must exist for the running interpreter and
  architecture. The checked in `emusFortran.cpython-310-x86_64-linux-gnu.so` is
  Python 3.10 on x86_64 only. On Apple Silicon, or any other Python version,
  rebuild it:

      python -m numpy.f2py -c bayflux/core/emusFortran.f90 -m emusFortran --backend meson

  f2py needs `meson`, `ninja` and a Fortran compiler on `PATH`, not merely
  installed in the virtualenv.

The simplest way to satisfy both is to run inside the container, which is also
how the release is gated:

    docker run --rm <dockerhub-user>/<image>:2.0 bash -c "cd /bayflux && python -m pytest tests -q"

To run the working copy rather than the image's own snapshot, mount it:

    docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work \
        <dockerhub-user>/<image>:2.0 python -m pytest tests -q

If a container built from `docker-compose.yml` is already up, exec into it:

    docker exec -it <container> bash -c "cd /bayflux && python3 -m pytest tests -q"

`release/build_and_publish.sh` runs the plain `pytest tests` form as the last
step of its `build` action.

## Selecting tests

Every test module is named `test_*.py`, so the suite collects correctly under
pytest's default settings, from any working directory, and under any other
project's configuration. This matters more than it looks: `setup.cfg` sets

    python_files = tests/*.py

and because that pattern contains a slash, pytest matches it against the whole
path and `*` crosses directory separators, so **every** `.py` file anywhere
under `tests/` is treated as a test module. The suite no longer depends on that
setting, but it is still in force, with two consequences:

- Data files that happen to end in `.py` are collected too and fail to import.
  `conftest.py` sets `collect_ignore_glob = ["fixtures/*"]` to keep the fixture
  directory out of collection, which is what makes
  `fixtures/toyModel/substrateLabelingDict.py` safe to check in. Put any new
  input data there.
- Helper modules cannot live loose under `tests/`; they would be imported as
  test modules. Put shared setup in `conftest.py`, which pytest handles
  separately and exempts from `python_files`.

Common selections:

    python -m pytest tests --collect-only -q              # what would run
    python -m pytest tests/core/test_sampler.py -v        # one file
    python -m pytest tests -k "acceptance or transition"  # by name
    python -m pytest tests -x -l                          # stop at first failure
    python -m pytest tests/core/test_steadyState.py::TestSteadyStateDistribution::testNormalization

## Measuring coverage

**A single coverage run gives the wrong answer for this package.** Coverage.py
cannot trace inside a Numba compiled function. With the JIT active,
`transitionProbabilities.py` reports 22 percent and `steadyState.py` 23 percent
while their tests pass. The reverse is also true: the `asFloatScalar` overload
selector in `likelihood.py` runs only while Numba is compiling, so it is
invisible when the JIT is disabled.

Take both runs and combine them:

    COVERAGE_FILE=.coverage.nojit NUMBA_DISABLE_JIT=1 \
        python -m pytest tests --cov=bayflux --cov-report=
    COVERAGE_FILE=.coverage.jit \
        python -m pytest tests --cov=bayflux --cov-report=
    python -m coverage combine .coverage.nojit .coverage.jit
    python -m coverage report --show-missing

Every number below comes from that combination. A plain
`python -m pytest tests --cov=bayflux` reports several points low for the same
test suite, for reasons that have nothing to do with the tests.

Other useful forms:

    python -m coverage html && open htmlcov/index.html    # read the branches
    python -m pytest tests --cov=bayflux.core.sampler --cov-report=term-missing

## Current coverage

    205 passed
    1573 statements, 40 missed, 97% covered

No skips and no expected failures: every test in the suite executes and passes.

| Module | Stmts | Cover | What is still missing |
| --- | --- | --- | --- |
| `bayflux/__init__.py` | 9 | 100% | |
| `core/__init__.py` | 4 | 100% | |
| `core/fluxes.py` | 30 | 100% | |
| `core/likelihood.py` | 53 | 100% | |
| `core/massDistribution.py` | 32 | 100% | |
| `core/sampler.py` | 77 | 100% | |
| `core/steadyState.py` | 44 | 100% | |
| `core/utilities.py` | 60 | 100% | |
| `core/worker.py` | 38 | 100% | |
| `core/emus.py` | 308 | 99% | sparse solver branch, line 821 |
| `core/reactionNetwork.py` | 246 | 99% | unreachable TypeError, line 108 |
| `core/findCenter.py` | 99 | 98% | sys.path guard and script entry, lines 27, 199 |
| `core/transitionProbabilities.py` | 41 | 98% | unreachable else, line 198 |
| `core/acceptanceRate.py` | 56 | 96% | two unreachable raises, lines 200, 202 |
| `core/bayflux.py` | 275 | 93% | dead resume, degenerate matrix handlers |
| `core/commandLine.py` | 124 | 88% | dead resume, MPI rank > 0 |



## What each file covers

| File | Covers |
| --- | --- |
| `core/test_emus.py` | EMU decomposition, compilation and simulation |
| `core/test_repeatMetabolite.py` | repeated metabolite EMU computation |
| `core/test_likelihood.py` | `normpdf` against scipy, edge cases, stability |
| `core/test_acceptanceRate.py` | acceptance rate arithmetic and burn-in |
| `core/test_steadyState.py` | power iteration, convergence, validation |
| `core/test_transitionProbabilities.py` | GMH transition matrix construction |
| `core/test_fluxes.py` | `FluxVector` construction, validation, display |
| `core/test_utilities.py` | resource logging, both precision helpers |
| `core/test_massDistribution.py` | MDV container, reader, writer, round trip |
| `core/test_acceptanceRateValidation.py` | input handling branches |
| `core/test_reactionNetworkIO.py` | constraint and transition file round trips |
| `core/test_reactionNetworkStructure.py` | gene, group, note and annotation copying |
| `core/test_likelihoodModel.py` | `logLikelihood` on the toy network |
| `core/test_emusExecution.py` | compiled operation executor input handling |
| `core/test_transitionProbabilitiesCorrection.py` | negative diagonal rescaling |
| `core/test_sampler.py` | `MCMCACHRSampler` centering and Metropolis phases |
| `core/test_samplerValidation.py` | reprojection and the validation retry loop |
| `core/test_commandLineEndToEnd.py` | `bayflux_serial` run end to end |
| `core/test_parallelSamplerEndToEnd.py` | `bayflux` run end to end |
| `core/test_findCenter.py` | `findCenter` run end to end, and its invocation |
| `core/test_worker.py` | the shared memory worker loop, in process |

### Testing the shared memory worker

`worker.py` was the largest uncovered file in the package, and its shape is why:
`samplerSharedMemory` is an unbounded `while True` with no exit condition, so
the parent process terminates it. It cannot be called from a test in the
ordinary way, and running it in a child process makes it invisible to coverage.

`test_worker.py` runs it in the pytest process and stops it through the one
seam the signature already provides, the `lock` argument. Every shared memory
access the worker makes happens inside `with lock`, so a lock object that
raises on entry ends the loop wherever the test chooses. Two details make that
sound rather than a trick:

- The stop signal subclasses `BaseException`, not `Exception`. The worker
  catches `Exception` around its sampling cycle, so an ordinary exception would
  be logged as a sampling failure and the loop would continue.
- The lock only interrupts when the worker's status flag already reads 2, which
  the worker sets as the last act of a cycle. At that moment it is back in its
  polling loop holding no state, so no cycle is ever cut in half.

Nothing is monkeypatched: the worker, the sampler and the shared memory layout
are the real ones, and only the lock and the process lifetime come from the
test. The nine tests cover a single cycle, repeated cycles, the polling loop,
the slot isolation that keeps a worker out of its neighbours' memory, the
failure path that writes `-inf` and still marks the worker processed, recovery
after a failed cycle, and the fact that each cycle re-reads the position from
shared memory rather than from its own last proposal.

If `samplerSharedMemory` ever gets a sentinel status that breaks its loop, as
suggested below, these tests should be rewritten to use it and the lock harness
retired.

## Fixtures

`conftest.py` builds the Antoniewicz 2006 Fig. 3 network in code, the same
network the shipped `command_line_example/test_toyModel` uses. It is small
enough that EMU compilation and simulation are effectively free.

| Fixture | What it gives you |
| --- | --- |
| `toyCobraModel` | the network as a plain `cobra.Model` |
| `toyNetwork` | as a `ReactionNetwork`, no atom transitions |
| `toyModel` | as a `ReactionNetwork`, with atom transitions |
| `toyFluxes` | a feasible `FluxVector` for it |
| `toySubstrateLabeling` | substrate A labeled on its second carbon |
| `toyOutputEmu` | the full size EMU of metabolite F |
| `toyCompiledData` | the EMU network compiled for that EMU |
| `toyMassDistribution` | experimental MDV for F matching the simulation |
| `toySampleableModel` | the network plus boundary reactions, for sampling |
| `toySampleableCobraModel` | the same, as a plain `cobra.Model` |
| `toyRunDirectory` | stages a complete input set and writes `config.yaml` |

`toyRunDirectory` returns a callable. Pass configuration overrides and it
writes `config.yaml` into a temporary directory, copies the five input files
from `fixtures/toyModel/` beside it, changes the working directory there and
returns the path:

    def testSomething(toyRunDirectory, monkeypatch):
        runDirectory = toyRunDirectory(outputBaseName="outputdata/toy",
                                       bayesianSamplesPerTask=5)
        monkeypatch.setattr(sys, "argv", ["bayflux_serial", "config.yaml"])
        ...

Both executables resolve their input paths relative to the working directory,
which is why the fixture changes it. Nothing is written inside `tests/` at run
time; every output lands in pytest's `tmp_path`.

The SBML model in `fixtures/toyModel/` is checked in rather than written by
cobra during the test, both to keep the end to end tests fast and to keep
libsbml's writer, which has segfaulted under some coverage configurations, off
the test path.

`fixtures/input_data/` holds the E. coli genome scale model, flux bounds, atom
transitions and GC-MS data, byte for byte the same files as
`command_line_example/test_ecoliGSM_NERSC/input_data/`. No test references it
yet. It is kept here for the one remaining reachable coverage gap, the sparse
solver branch at `emus.py:821`, which needs an EMU matrix of more than 100
rows. Delete it if that test is not going to be written; do not add a second
copy of any large model.

`conftest.py` also does `import bayflux` before anything else. That applies the
numpy compatibility shim in `bayflux/__init__.py` (`np.object = object`) which
the pinned COBRApy fork still needs. Without it, any module that imports cobra
first fails to import under numpy 1.24 or newer. Be aware that this hides
import order problems from every in-process test; `test_findCenter.py` uses
subprocesses where that distinction matters.

## Defects this suite found, and where they were fixed

The suite carried four `xfail(strict=True)` markers, each documenting a real
defect so that the defective lines still counted toward coverage and the test
turned into a failure the moment the defect was fixed. All four are now fixed
and no expected failures remain.

1. **`writeFluxConstraints(exchangesOnly=False)` raised on any model with more
   reactions than exchanges.** It built its rows from `self.exchanges` but its
   index from the selected reactions. `reactionNetwork.py` now builds both from
   `reactions`, which is identical on the `exchangesOnly=True` default path.
2. **A group containing another group could not be copied.** The nested `Group`
   branch of the member remapping looked the member up in `self.genes` instead
   of `self.groups`. The two-pass construction already populates `self.groups`
   before members are remapped, so the corrected lookup always resolves.
3. **Zero-dimensional sub-arrays raised the wrong error.** `len()` on a
   zero-dimensional array raises `TypeError`, which escaped
   `computeAcceptanceRate` before its intended `ValueError`. The guard now
   tests `ndim` first. Zero-dimensional arrays are easy to produce from a
   notebook by indexing a numpy or pandas object.
4. **`findCenter.py` could not be run by path.** Python puts a script's own
   directory at the front of `sys.path`, ahead of `PYTHONPATH`, so
   `bayflux/core/bayflux.py` satisfied `import bayflux` before the package did
   and `bayflux/__init__.py` never applied the numpy shim. `findCenter.py` now
   drops that entry when, and only when, it is the script directory.
   `test_findCenter.py` covers the guard from three directions: by path works,
   `-m` works, and an ordinary import leaves the caller's `sys.path[0]` alone.
   The companion test `testImportingBayfluxFromInsideCoreResolvesToTheModule`
   pins the underlying name collision; if `bayflux/core/bayflux.py` is ever
   renamed, that test and the guard retire together.

Two further defects are documented in the code rather than as tests, because
they produce no observable failure:

- **Resume from previous output can never trigger**, in either executable. Both
  scan `os.listdir()` of the working directory for names matching a pattern
  built from `outputBaseName`. That value must contain a directory component,
  because `ntpath.dirname` of a bare name gives `''` and `os.mkdir('')` raises
  `FileNotFoundError`, which is not caught. So the pattern always carries a
  `/` and can never match a bare `listdir` entry.

## What is not covered, and why

Forty statements remain uncovered. Most cannot
be reached at all:

- `transitionProbabilities.py:198` sets the diagonal to zero when it is negative
  but the off-diagonal sum is at most one. The diagonal is `1 - offDiagonalSum`,
  so that combination is impossible. Dead defensive code.
- `acceptanceRate.py:200,202` are a zero-dimensional branch that the `ndim`
  guard now reaches first, and a `could not broadcast` branch whose message
  numpy no longer produces; ragged input now says `inhomogeneous shape` and
  lands in the generic handler.
- `reactionNetwork.py:108` raises `TypeError` for a group member that is none of
  metabolite, reaction, gene or group. COBRApy's API cannot produce one.
- `findCenter.py:27` is the `sys.path` guard, which fires only when the file is
  run as a script. The test that exercises it runs a subprocess, which coverage
  does not trace. `findCenter.py:199` is `if __name__ == "__main__": main()`.
- `commandLine.py:79-81,127-129,179-182,193,201-204` are the dead resume paths
  and the MPI send and receive branches for rank greater than zero. Covering
  the latter needs a test that launches `mpirun -n 2` as a subprocess, which
  contributes nothing to coverage without `COVERAGE_PROCESS_START` plumbing.
- `bayflux.py:287-289,551,553,590-604,612,734` handle a transition matrix with
  negative entries or rows that do not sum to one, plus the same dead resume
  path. Reaching them needs a deliberately degenerate matrix, which the input
  validation in `transitionProbabilities.py` makes harder to produce by
  accident.
- `emus.py:821` selects the sparse solver for EMU matrices with more than 100
  rows. The genome scale model in `fixtures/input_data/` would cover it, at the
  cost of a slower test. This is the one remaining reachable gap, worth roughly
  one percentage point.

**Realistic ceiling.** Adding a genome scale test for the sparse solver branch
leaves nothing else reachable, which is already close to the 97 percent
reported above. One hundred percent is not a sensible target here: several
remaining statements are provably unreachable, and some are unreachable because
of bugs that should be fixed rather than tested around.

Worth noting for anyone reading the numbers later: `worker.py` went from 18
percent to 100 percent with no change to `bayflux/`, so an unbounded loop in a
child process is not by itself a reason to leave a module untested. The
sentinel-status change suggested for `samplerSharedMemory` is still worth
making, but for clean worker shutdown rather than for coverage.

## Warnings

A clean run emits about thirty warnings, none of them failures. They come from
three places:

- **Dependencies.** SWIG type warnings from `python-libsbml`, a
  `np.find_common_type` deprecation from pandas, and COBRApy's own notice about
  reactions constrained to a fixed flux value. Nothing in this repository can
  fix these without changing the pins.
- **The package's own `UserWarning`** from `acceptanceRate.py`, raised when
  fewer than the minimum number of samples remain after burn-in. The end to end
  tests deliberately run tiny chains, so they trigger it on purpose.
- **`commandLine.py:71`** raises `DeprecationWarning: invalid escape sequence
  '\d'` the first time the module is byte-compiled. The regular expression
  there needs an `r''` prefix. It is a one-character source fix, outside this
  directory.

## Adding tests

- Name new files `test_<subject>.py` under `core/`. Any `.py` file under
  `tests/` outside `fixtures/` is collected, so do not put helpers there;
  put shared setup in `conftest.py`, which is exempt.
- Put input data under `fixtures/`, which is excluded from collection, and keep
  it small. Large models already in the repository should be referenced from
  `command_line_example/`, not copied here.
- Use the toy network fixtures rather than building a model inline. If a test
  needs to sample, use `toySampleableModel`, which has the boundary reactions
  that make the network solvable; without them every internal flux is pinned to
  zero at steady state.
- Write to `tmp_path`, never into the repository, and never hardcode an absolute
  path or a hostname.
- Keep assertions inside test functions and return nothing from them. A test
  that returns a value is a pytest warning today and an error in a future
  release; if module level code needs the value, compute it at module level.
- Keep function names camelCase, matching the rest of the codebase.
- If a test would document a defect rather than a requirement, mark it
  `xfail(strict=True)` with a reason that names the cause, so it converts into a
  signal when the defect is fixed.
- Re-measure with the two-run combination above, not a single `--cov` run.


