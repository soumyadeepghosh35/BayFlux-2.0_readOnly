"""The validatecheck retry loop and the reprojection branch of MCMCACHRSampler.

Both are hard to reach with ordinary inputs. The retry loop only runs when
cobra's validate returns a code that is not a substring of 'v', and
reprojection only runs when the sampling problem is homogeneous, which
excludes any model with a reaction pinned to a fixed non-zero flux.
"""

import numpy as np
import pytest

import bayflux
from bayflux.core.sampler import MCMCACHRSampler


@pytest.fixture
def toyHomogeneousModel(toySampleableModel):
    """The toy network with no fixed non-zero bound, so the problem is homogeneous.

    A reaction whose lower and upper bounds are equal and non-zero becomes an
    equality with a non-zero right hand side, which makes the problem
    inhomogeneous and disables reprojection.
    """
    toySampleableModel.reactions.get_by_id("a_exchange").bounds = (-100.0, -50.0)
    return toySampleableModel


def testHomogeneousProblemsAreReprojected(toyHomogeneousModel):
    sampler = MCMCACHRSampler(toyHomogeneousModel, thinning=2, seed=5)
    assert sampler.problem.homogeneous, \
        "fixture no longer produces a homogeneous problem"

    reprojections = []
    originalReproject = sampler._reproject

    def countingReproject(vector):
        reprojections.append(1)
        return originalReproject(vector)

    sampler._reproject = countingReproject
    samples = sampler.sample(3)

    assert reprojections, "reprojection never ran on a homogeneous problem"
    assert np.isfinite(samples.to_numpy()).all()


def testInvalidSamplesTriggerARetry(toySampleableModel, capsys):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=1, seed=5)

    # 'lb' is not a substring of 'v', so the guard treats it as invalid and
    # the loop searches for another sample. An empty code ends the search.
    verdicts = [np.array(["lb"]), np.array(["lb"]), np.array([""])]

    def stubbornValidate(samples, **kwargs):
        return verdicts.pop(0) if verdicts else np.array([""])

    sampler.validate = stubbornValidate
    samples = sampler.sample(1, validatecheck=True)

    assert samples.shape[0] == 1
    assert "searching new valid sample" in capsys.readouterr().out


def testGivingUpOnValidationExitsTheProcess(toySampleableModel, capsys):
    sampler = MCMCACHRSampler(toySampleableModel, thinning=1, seed=5)

    def alwaysInvalid(samples, **kwargs):
        return np.array(["lb"])

    sampler.validate = alwaysInvalid

    # after ten failed attempts the sampler calls sys.exit, which a caller can
    # only observe as SystemExit
    with pytest.raises(SystemExit):
        sampler.sample(1, validatecheck=True)

    assert "Tried to find valid sample" in capsys.readouterr().out
