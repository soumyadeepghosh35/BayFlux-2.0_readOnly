"""Tests for bayflux.core.worker, the shared memory sampling worker.

`samplerSharedMemory` is an unbounded `while True` with no exit condition: in
production the parent process terminates it. That makes it impossible to call
from a test in the ordinary way, and invisible to coverage if it is run in a
child process, which is why this module was the largest uncovered file in the
package.

These tests run the worker in this process and stop it through the one seam it
already has: the lock it is handed. Every shared memory access the worker makes
happens inside `with lock`, so a lock object that raises on entry ends the loop
at a point of our choosing. Two details make that safe:

- The stop signal derives from BaseException rather than Exception, so the
  worker's own `except Exception` handler cannot swallow it and the function
  unwinds instead of starting another iteration.
- The lock only interrupts when the worker has already marked itself processed,
  which means it is back in its polling loop holding no state. Interrupting
  mid-cycle would prove nothing about the cycle.

Nothing here monkeypatches the worker or the sampler. The code under test is
the same code the parallel executable runs; only the lock and the process
lifetime are supplied by the test.
"""

import logging
import threading

import numpy as np
import pytest

from bayflux.core.sampler import MCMCACHRSampler
from bayflux.core.worker import samplerSharedMemory


try:
    from multiprocessing import shared_memory
    _probe = shared_memory.SharedMemory(create=True, size=8)
    _probe.close()
    _probe.unlink()
except Exception as error:  # pragma: no cover
    pytest.skip(
        "POSIX shared memory is unavailable: %s" % error, allow_module_level=True
    )


WORKER_ID = 1
NUM_PROPOSALS = 2


class StopWorker(BaseException):
    """Signal that ends the worker loop.

    Deliberately not an Exception. The worker catches Exception around its
    sampling cycle and would otherwise treat the stop signal as a sampling
    failure and keep going.
    """


class CycleLimitedLock:
    """A real lock that stops the worker after a fixed number of cycles.

    The worker enters the lock three times per successful cycle: once per poll
    of its status flag, once to copy the current position, and once to write
    its results back. Rather than counting those, this class watches the status
    flag: the worker sets it to 2 as the last act of a cycle, so seeing 2 on
    entry means one full cycle just finished and the worker is polling again.
    At that point the lock either re-arms the flag for another cycle or raises.
    """

    def __init__(self, statusShared, cycles):
        self.statusShared = statusShared
        self.cycles = cycles
        self.cyclesCompleted = 0
        self._lock = threading.Lock()

    def __enter__(self):
        if self.statusShared[WORKER_ID] == 2:
            self.cyclesCompleted += 1
            if self.cyclesCompleted >= self.cycles:
                raise StopWorker
            self.statusShared[WORKER_ID] = 1
        self._lock.acquire()
        return self._lock

    def __exit__(self, exceptionType, exceptionValue, traceback):
        self._lock.release()
        return False


class WorkerHarness:
    """The shared memory layout bayflux.core.bayflux builds for its workers."""

    def __init__(self, model):
        self.model = model
        self.fluxVectorSize = len(model.variables)

        sampler = MCMCACHRSampler(model, thinning=1, seed=11)
        self.center = sampler.center.copy()
        self.startingPosition = sampler.prev.copy()

        itemSize = np.float64().nbytes
        self.blocks = []
        self.positionShm = self._create(self.fluxVectorSize * itemSize)
        self.proposalsShm = self._create(
            (NUM_PROPOSALS + 1) * self.fluxVectorSize * itemSize
        )
        self.likelihoodsShm = self._create((NUM_PROPOSALS + 1) * itemSize)
        self.statusShm = self._create((NUM_PROPOSALS + 1) * np.int32().nbytes)

        self.position = np.ndarray(
            (self.fluxVectorSize,), dtype=np.float64, buffer=self.positionShm.buf
        )
        self.proposals = np.ndarray(
            (NUM_PROPOSALS + 1, self.fluxVectorSize),
            dtype=np.float64, buffer=self.proposalsShm.buf,
        )
        self.likelihoods = np.ndarray(
            (NUM_PROPOSALS + 1,), dtype=np.float64, buffer=self.likelihoodsShm.buf
        )
        self.status = np.ndarray(
            (NUM_PROPOSALS + 1,), dtype=np.int32, buffer=self.statusShm.buf
        )

        self.proposals.fill(0.0)
        self.likelihoods.fill(0.0)
        self.status.fill(0)
        self.position[:] = self.startingPosition

    def _create(self, size):
        block = shared_memory.SharedMemory(create=True, size=size)
        self.blocks.append(block)
        return block

    def run(self, logLikelihood, cycles=1, seed=7):
        """Drive the worker for a fixed number of cycles, then let it unwind.

        Returns the lock, whose cyclesCompleted count says how many cycles the
        worker actually finished.
        """
        self.status[WORKER_ID] = 1  # the parent's signal to start processing
        lock = CycleLimitedLock(self.status, cycles)
        with pytest.raises(StopWorker):
            samplerSharedMemory(
                WORKER_ID,
                seed,
                self.model,
                self.center,
                logLikelihood,
                self.positionShm.name,
                self.proposalsShm.name,
                self.likelihoodsShm.name,
                self.statusShm.name,
                self.fluxVectorSize,
                NUM_PROPOSALS,
                lock,
            )
        return lock

    def close(self):
        for block in self.blocks:
            block.close()
            block.unlink()


@pytest.fixture
def workerHarness(toySampleableModel):
    harness = WorkerHarness(toySampleableModel)
    try:
        yield harness
    finally:
        # release the numpy views before the buffers go away, or BufferError
        harness.position = harness.proposals = None
        harness.likelihoods = harness.status = None
        harness.close()


class RecordingLikelihood:
    """A log-likelihood that records what it was asked to score."""

    def __init__(self, value=-12.5):
        self.value = value
        self.arguments = []

    def __call__(self, fluxVector):
        self.arguments.append(np.asarray(fluxVector).copy())
        return self.value


def testWorkerWritesOneProposalAndItsLikelihood(workerHarness):
    likelihood = RecordingLikelihood()

    workerHarness.run(likelihood, cycles=1)

    assert len(likelihood.arguments) == 1
    assert workerHarness.status[WORKER_ID] == 2
    assert workerHarness.likelihoods[WORKER_ID] == likelihood.value
    # the proposal written to shared memory is the vector that was scored
    np.testing.assert_array_equal(
        workerHarness.proposals[WORKER_ID], likelihood.arguments[0]
    )


def testWorkerProposalIsAFiniteMoveFromTheSharedPosition(workerHarness):
    likelihood = RecordingLikelihood()

    workerHarness.run(likelihood, cycles=1)

    proposal = workerHarness.proposals[WORKER_ID]
    assert proposal.shape == (workerHarness.fluxVectorSize,)
    assert np.isfinite(proposal).all()
    # ACHR takes a step, so the proposal cannot be the position it started from
    assert not np.array_equal(proposal, workerHarness.startingPosition)
    # and it must respect the flux bounds, or the sampler has broken its contract
    assert (proposal >= -1e-9).all()


def testWorkerOnlyWritesItsOwnSlot(workerHarness):
    workerHarness.run(RecordingLikelihood(), cycles=1)

    # slot 0 belongs to the current position and the other worker's slot to a
    # process that never ran, so both must still hold their initial zeros
    for slot in (0, WORKER_ID + 1):
        np.testing.assert_array_equal(
            workerHarness.proposals[slot],
            np.zeros(workerHarness.fluxVectorSize),
        )
        assert workerHarness.likelihoods[slot] == 0.0
        assert workerHarness.status[slot] == 0


def testWorkerKeepsSamplingAcrossCycles(workerHarness):
    likelihood = RecordingLikelihood()

    lock = workerHarness.run(likelihood, cycles=3)

    assert lock.cyclesCompleted == 3
    assert len(likelihood.arguments) == 3
    assert workerHarness.status[WORKER_ID] == 2
    # each cycle re-reads the shared position, which the test never changes, so
    # the chain must still advance under its own randomness
    assert not np.array_equal(likelihood.arguments[0], likelihood.arguments[-1])


def testWorkerReadsThePositionFromSharedMemory(workerHarness):
    """The proposal must depend on what the parent last wrote to shared memory.

    With the worker's seed and centre held fixed, a cycle is deterministic, so
    the shared position is the only thing left that can change the proposal.
    Repeating a cycle unchanged must reproduce it exactly; changing only the
    shared position must not.
    """
    first = RecordingLikelihood()
    workerHarness.run(first, cycles=1, seed=5)

    repeated = RecordingLikelihood()
    workerHarness.position[:] = workerHarness.startingPosition
    workerHarness.run(repeated, cycles=1, seed=5)
    np.testing.assert_allclose(repeated.arguments[0], first.arguments[0])

    # the worker's own validated proposal is a feasible interior point, so it
    # makes a legitimate new position without the test having to construct one
    otherPosition = first.arguments[0]
    assert not np.array_equal(otherPosition, workerHarness.startingPosition)

    moved = RecordingLikelihood()
    workerHarness.position[:] = otherPosition
    workerHarness.run(moved, cycles=1, seed=5)
    assert not np.array_equal(moved.arguments[0], first.arguments[0])


def testWorkerRecordsNegativeInfinityWhenLikelihoodFails(workerHarness, caplog):
    def failingLikelihood(fluxVector):
        raise RuntimeError("measurement model exploded")

    with caplog.at_level(logging.ERROR):
        workerHarness.run(failingLikelihood, cycles=1)

    assert workerHarness.likelihoods[WORKER_ID] == float("-inf")
    # still marked processed, otherwise the parent waits forever
    assert workerHarness.status[WORKER_ID] == 2
    assert "Worker %d encountered error during sampling" % WORKER_ID in caplog.text
    assert "measurement model exploded" in caplog.text


def testWorkerLeavesNoProposalBehindWhenTheCycleFails(workerHarness):
    def failingLikelihood(fluxVector):
        raise RuntimeError("measurement model exploded")

    workerHarness.run(failingLikelihood, cycles=1)

    # the failure path writes the likelihood and the status but never the
    # proposal, so slot 1 keeps its zeros. The parent scores it as -inf, which
    # is why leaving stale flux values there does not corrupt the chain.
    np.testing.assert_array_equal(
        workerHarness.proposals[WORKER_ID], np.zeros(workerHarness.fluxVectorSize)
    )


def testWorkerRecoversAfterAFailedCycle(workerHarness):
    calls = {"count": 0}

    def intermittentLikelihood(fluxVector):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return -3.25

    workerHarness.run(intermittentLikelihood, cycles=2)

    assert calls["count"] == 2
    # the second cycle succeeded, so the -inf from the first is overwritten
    assert workerHarness.likelihoods[WORKER_ID] == -3.25
    assert np.isfinite(workerHarness.proposals[WORKER_ID]).all()


def testWorkerWaitsWhileItsStatusFlagIsNotSet(workerHarness):
    """Cover the polling loop, which is the only part that can spin.

    The worker polls until its flag reads 1. A lock that flips the flag on the
    third poll proves the loop actually waits rather than falling through.
    """
    class DelayedStartLock(CycleLimitedLock):
        def __init__(self, statusShared, cycles, pollsBeforeReady):
            super().__init__(statusShared, cycles)
            self.polls = 0
            self.pollsBeforeReady = pollsBeforeReady

        def __enter__(self):
            if self.statusShared[WORKER_ID] == 0:
                self.polls += 1
                if self.polls >= self.pollsBeforeReady:
                    self.statusShared[WORKER_ID] = 1
            return super().__enter__()

    workerHarness.status[WORKER_ID] = 0  # not ready yet
    lock = DelayedStartLock(workerHarness.status, cycles=1, pollsBeforeReady=3)
    with pytest.raises(StopWorker):
        samplerSharedMemory(
            WORKER_ID, 7, workerHarness.model, workerHarness.center,
            RecordingLikelihood(), workerHarness.positionShm.name,
            workerHarness.proposalsShm.name, workerHarness.likelihoodsShm.name,
            workerHarness.statusShm.name, workerHarness.fluxVectorSize,
            NUM_PROPOSALS, lock,
        )

    assert lock.polls == 3
    assert workerHarness.status[WORKER_ID] == 2
