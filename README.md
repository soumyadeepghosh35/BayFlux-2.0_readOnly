# BayFlux-2.0 (read-only code review copy)

This repository is a curated excerpt of BayFlux 2.0, published so that readers can inspect the code I wrote. It is read-only and incomplete by design. It cannot be installed or run from this repository alone, and it contains no models, data or results. The runnable version is a private container, available on request (see "Running the code with Docker" below). The manuscript describing BayFlux 2.0 is in preparation.

## What BayFlux is

BayFlux is a Bayesian method for 13C metabolic flux analysis (13C-MFA) on genome-scale metabolic models. It returns the full distribution of fluxes compatible with isotope labeling data, with uncertainty traceable to the measurements. The method and its validation are described in Backman et al., PLOS Computational Biology 2023 (https://doi.org/10.1371/journal.pcbi.1011111). That paper covers BayFlux 1.0, which uses a single Markov chain.

BayFlux 2.0 replaces the single chain with multiple-proposal MCMC (MP-MCMC) and runs the proposals in parallel on one node.

## How the parallel sampler works

The flow below follows `bayflux/core/bayflux.py` and `bayflux/core/worker.py`.

1. The parent process reads a YAML configuration, loads the model with COBRApy, and compiles the elementary metabolite unit (EMU) network for the labeling simulation.
2. It creates shared memory blocks for the current flux position, for N+1 candidate flux vectors (the current position plus N proposals), for their log-likelihoods, and for a status flag per worker.
3. It starts N worker processes. Each worker draws a proposal from an artificially centered hit-and-run (ACHR) sampler, evaluates the log-likelihood of that proposal against the labeling data, and writes both to shared memory.
4. When all workers are done, the parent builds a transition probability matrix over the N+1 candidates and uses it to select the next samples. Then the workers start the next round from the new position.
5. Acceptance rate is computed from the accepted samples after the run.

Shared memory avoids pickling flux vectors between processes. Each process is limited to one BLAS thread, so that parallelism comes from the proposals and not from nested threading.

## Where to look

| Path | What it holds |
| --- | --- |
| `bayflux/core/bayflux.py` | Entry point of the parallel sampler: configuration, setup, shared memory, worker launch, main sampling loop |
| `bayflux/core/worker.py` | Worker process that proposes a flux vector and scores it |
| `bayflux/core/transitionProbabilities.py` | Transition probabilities over the N+1 candidates |
| `bayflux/core/sampler.py` | ACHR sampler extended with a likelihood and a Metropolis test |
| `bayflux/core/likelihood.py` | Log-likelihood of a flux vector against measured mass distributions |
| `bayflux/core/steadyState.py` | Steady-state distribution used by the transition step |
| `bayflux/core/findCenter.py` | Centering of the sampler before the likelihood run |
| `bayflux/core/acceptanceRate.py` | Acceptance rate with checks on short chains |
| `bayflux/core/utilities.py` | Resource logging and numerical precision checks |
| `tests/` | Unit and end-to-end tests |
| `notebooks/` | Analysis and demonstration notebooks, with outputs cleared |
| `reference/ci-cd/` | Continuous integration workflows for the container images, names redacted |
| `docs/theory.rst` | Theory page of the documentation |

The EMU forward model (`emus.py`, `reactionNetwork.py`, `fluxes.py`, `massDistribution.py`) comes from BayFlux 1.0 and was written by Tyler Backman. I made smaller changes to it.

## What is not included

- The compiled EMU kernel and its Fortran source.
- Metabolic models, flux bounds, labeling data, sample outputs and test fixtures.
- Packaging files, dependency pins, Dockerfiles, entry point scripts and container image names.
- Benchmark results and manuscript figures.
- Unreleased components.

Without these, the code can be read and reviewed, but not executed.

## Running the code with Docker

The runnable version of BayFlux 2.0 is distributed as a Docker container. It holds the compiled EMU kernel, the Python environment, the example models and data, and the demonstration notebooks. The container is private and available upon request. When access is granted, I send the image names and the pull credentials. Nothing in this repository is enough to run the code without it.

You need Docker, about 16 GB of free disk for the image, and at least 8 GB of memory given to Docker Desktop on a Mac. The parallel sampler runs one worker per proposal, so more CPU cores help.

There are two images, one per processor type. Use `<image>:2.0` on Linux x86-64, Intel Macs and Windows. Use `<image>:2.0-arm64` on Apple Silicon Macs, where the x86-64 image would run only under emulation.

| Step | Action | Command |
| --- | --- | --- |
| 1 | Log in and pull the image | `docker login`<br>`docker pull <image>:2.0` |
| 2 | Check that it starts | `docker run --rm <image>:2.0 bayflux --help` |
| 3 | Copy the examples into a working folder | `mkdir myrun && cd myrun`<br>`docker run --rm -v "$PWD":/work <image>:2.0 bayflux-copy-examples /work` |
| 4 | Run the toy model | `cd command_line_example/test_toyModel`<br>`docker run --rm --shm-size=4g -v "$PWD":/work <image>:2.0 bayflux config.yaml` |

The toy model takes about one minute. Results are written into the folder you mounted at `/work`, so they appear on your machine. Always pass `--shm-size=4g`, because the sampler shares memory between processes and long runs can fail with `No space left on device` without it.

To run a genome-scale example, copy the examples as in step 3 and run `bayflux config.yaml` from `command_line_example/test_ecoliGSM/singleChain` in the same way. The single chain example takes roughly 5 to 10 minutes on a workstation. Add `--all` to the copy command in step 3 to also copy the notebooks and the test suite.

## Copyright and license

Copyright remains with the holders named in `license.txt`. That file is reproduced unchanged. This repository is a partial copy published for review of my work, and it is not a distribution of BayFlux.
