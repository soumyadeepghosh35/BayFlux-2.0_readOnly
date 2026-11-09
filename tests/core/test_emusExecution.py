"""Input handling in bayflux.core.emus.executeCompiledEMUTransitionList.

The compiled operation executor accepts either a raw numpy flux array or a
FluxVector. Only the array path was covered.
"""

import numpy as np

import bayflux
from bayflux.core.emus import (
    compileEMUTransitionList,
    constructMatrixCoords,
    constructZeroMatrices,
    emuDecomposition,
    executeCompiledEMUTransitionList,
    splitBySize,
)


def compileSizeOneOperations(model, outputEmu):
    transitions = emuDecomposition([outputEmu])
    transitionsBySize = splitBySize(transitions)
    matrixCoords = constructMatrixCoords(transitionsBySize[1])
    operations = compileEMUTransitionList(transitionsBySize[1], matrixCoords, model)
    zeroMatrices = constructZeroMatrices(matrixCoords)
    return operations, zeroMatrices


def testFluxVectorAndRawArrayGiveTheSameMatrices(
    toyModel, toyOutputEmu, toyFluxes
):
    operations, zeroMatrices = compileSizeOneOperations(toyModel, toyOutputEmu)

    fromVector = executeCompiledEMUTransitionList(
        operations, zeroMatrices, toyFluxes
    )
    fromArray = executeCompiledEMUTransitionList(
        operations, zeroMatrices, toyFluxes.fluxes
    )

    assert len(fromVector) == len(fromArray) == 2
    for vectorMatrix, arrayMatrix in zip(fromVector, fromArray):
        assert np.array_equal(vectorMatrix, arrayMatrix)


def testZeroMatricesAreNotModifiedInPlace(toyModel, toyOutputEmu, toyFluxes):
    operations, zeroMatrices = compileSizeOneOperations(toyModel, toyOutputEmu)
    before = [matrix.copy() for matrix in zeroMatrices]

    executeCompiledEMUTransitionList(operations, zeroMatrices, toyFluxes)

    for original, current in zip(before, zeroMatrices):
        assert np.array_equal(original, current), \
            "the executor must copy its input matrices rather than overwrite them"
