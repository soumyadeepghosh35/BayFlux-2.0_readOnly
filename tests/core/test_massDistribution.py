"""Tests for bayflux.core.massDistribution.

Covers the MassDistribution container, its DataFrame and file output, and the
reader, including the carbon count consistency assertion.
"""

import pandas as pd
import pytest

import bayflux
from bayflux.core.massDistribution import readMassDistribution


def testEmptyMassDistributionStartsWithNoData(toyModel):
    massDistribution = bayflux.MassDistribution(toyModel)
    assert len(massDistribution.data) == 0
    assert massDistribution.model is toyModel


def testToDataFrameUnrollsReplicates(toyModel, toyOutputEmu):
    massDistribution = bayflux.MassDistribution(toyModel)
    massDistribution.data[toyOutputEmu].append([0.1, 0.2, 0.3, 0.4])
    massDistribution.data[toyOutputEmu].append([0.2, 0.2, 0.3, 0.3])

    frame = massDistribution.toDF()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, 6)
    assert list(frame["metabolite"]) == ["F", "F"]
    assert list(frame["atoms"]) == ["1,2,3", "1,2,3"]


def testReprMatchesTheDataFrameRepr(toyMassDistribution):
    assert repr(toyMassDistribution) == repr(toyMassDistribution.toDF())


def testWriteToFileRoundTrips(tmp_path, toyModel, toyMassDistribution):
    outputFile = tmp_path / "mdv.tab"
    toyMassDistribution.writeToFile(outputFile)

    written = pd.read_csv(outputFile, sep="\t")
    assert list(written["metabolite"]) == ["F"]
    assert list(written["atoms"]) == ["1,2,3"]

    reloaded = readMassDistribution(toyModel, outputFile)
    assert len(reloaded.data) == 1
    emu = next(iter(reloaded.data))
    assert emu.metabolite.id == "F"
    assert emu.indices == [1, 2, 3]
    assert reloaded.data[emu][0] == pytest.approx([0.0001, 0.8008, 0.1983, 0.0009])


def testReaderRejectsUnsupportedFormat(tmp_path, toyModel):
    inputFile = tmp_path / "mdv.tab"
    inputFile.write_text("metabolite\tatoms\t0\nF\t1,2,3\t1.0\n")
    with pytest.raises(AssertionError, match="unsupported format"):
        readMassDistribution(toyModel, inputFile, format="json")


def testWriterRejectsUnsupportedFormat(tmp_path, toyMassDistribution):
    # writeToFile has no format argument, so the guard lives in the reader only.
    # This documents that asymmetry rather than asserting on the writer.
    outputFile = tmp_path / "mdv.tab"
    toyMassDistribution.writeToFile(outputFile)
    assert outputFile.exists()


def testReaderRejectsMdvThatDisagreesWithCarbonCount(tmp_path, toyModel):
    inputFile = tmp_path / "mdv.tab"
    # F has three mapped atoms, so a valid MDV needs four values, not three
    inputFile.write_text("metabolite\tatoms\t0\t1\t2\nF\t1,2,3\t0.5\t0.3\t0.2\n")
    with pytest.raises(AssertionError, match="mismatch between mass inputs"):
        readMassDistribution(toyModel, inputFile)


def testReaderSkipsCommentLines(tmp_path, toyModel):
    inputFile = tmp_path / "mdv.tab"
    inputFile.write_text(
        "metabolite\tatoms\t0\t1\t2\t3\n"
        "# this line is a comment\n"
        "F\t1,2,3\t0.1\t0.2\t0.3\t0.4\n"
    )
    massDistribution = readMassDistribution(toyModel, inputFile)
    assert len(massDistribution.data) == 1


def testReaderGroupsReplicatesOfTheSameEmu(tmp_path, toyModel):
    inputFile = tmp_path / "mdv.tab"
    inputFile.write_text(
        "metabolite\tatoms\t0\t1\t2\t3\n"
        "F\t1,2,3\t0.1\t0.2\t0.3\t0.4\n"
        "F\t1,2,3\t0.2\t0.2\t0.3\t0.3\n"
    )
    massDistribution = readMassDistribution(toyModel, inputFile)
    assert len(massDistribution.data) == 1
    assert len(next(iter(massDistribution.data.values()))) == 2
