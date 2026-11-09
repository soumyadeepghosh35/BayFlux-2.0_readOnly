"""File round trips and copying for bayflux.core.reactionNetwork.

The reader and writer pairs for flux constraints and atom transitions carry
most of the validation logic in this module and none of it was exercised.
"""

import pandas as pd
import pytest

import bayflux


def testReactionNetworkWrapsReactionsAsEnhancedReactions(toyNetwork):
    assert all(isinstance(reaction, bayflux.EnhancedReaction)
               for reaction in toyNetwork.reactions)
    assert all(reaction.transitions == [] for reaction in toyNetwork.reactions)


def testCopyIsIndependentOfTheOriginal(toyModel):
    duplicate = toyModel.copy()

    assert isinstance(duplicate, bayflux.ReactionNetwork)
    assert len(duplicate.reactions) == len(toyModel.reactions)
    assert len(duplicate.metabolites) == len(toyModel.metabolites)
    assert all(isinstance(reaction, bayflux.EnhancedReaction)
               for reaction in duplicate.reactions)

    duplicate.reactions.get_by_id("a_b").upper_bound = 123.0
    assert toyModel.reactions.get_by_id("a_b").upper_bound == 500.0


def testCopyCarriesAtomTransitions(toyModel):
    duplicate = toyModel.copy()
    assert repr(duplicate.reactions.get_by_id("b_d").transitions[0]) == \
        repr(toyModel.reactions.get_by_id("b_d").transitions[0])


def testEnhancedReactionHtmlIncludesTransitions(toyModel):
    html = toyModel.reactions.get_by_id("a_b")._repr_html_()
    assert "a_b" in html
    assert "abc : abc" in html


def testAtomTransitionReverseSwapsReactantsAndProducts(toyModel):
    forward = toyModel.reactions.get_by_id("a_b").transitions[0]
    reverse = forward.reverse()
    assert repr(forward) == "A --> B\tabc : abc"
    assert repr(reverse) == "B --> A\tabc : abc"


def testFluxConstraintRoundTrip(tmp_path, toyModel):
    boundsFile = tmp_path / "bounds.csv"
    toyModel.writeFluxConstraints(boundsFile)

    written = pd.read_csv(boundsFile, index_col=0)
    assert list(written.columns) == ["lower_bound", "upper_bound"]

    # the toy network has no boundary reactions, so the export is empty
    assert written.shape[0] == len(toyModel.exchanges)


def testFluxConstraintsCanBeAppliedFromFile(tmp_path, toyModel):
    boundsFile = tmp_path / "bounds.csv"
    boundsFile.write_text(
        ",lower_bound,upper_bound\n"
        "a_b,10.0,90.0\n"
        "b_d,-20.0,20.0\n"
    )
    toyModel.readFluxConstraints(boundsFile)

    assert toyModel.reactions.get_by_id("a_b").lower_bound == 10.0
    assert toyModel.reactions.get_by_id("a_b").upper_bound == 90.0
    assert toyModel.reactions.get_by_id("b_d").lower_bound == -20.0


def testFluxConstraintReaderRejectsInvertedBounds(tmp_path, toyModel):
    boundsFile = tmp_path / "bounds.csv"
    boundsFile.write_text(",lower_bound,upper_bound\na_b,90.0,10.0\n")
    with pytest.raises(AssertionError, match="larger than upper bound"):
        toyModel.readFluxConstraints(boundsFile)


def testFluxConstraintReaderRejectsUnknownReaction(tmp_path, toyModel):
    boundsFile = tmp_path / "bounds.csv"
    boundsFile.write_text(",lower_bound,upper_bound\nnot_a_reaction,0.0,1.0\n")
    with pytest.raises(AssertionError, match="missing from self"):
        toyModel.readFluxConstraints(boundsFile)


def testFluxConstraintReaderRejectsUnsupportedFormat(tmp_path, toyModel):
    with pytest.raises(AssertionError, match="unsupported format"):
        toyModel.readFluxConstraints(tmp_path / "bounds.csv", format="json")


def testFluxConstraintWriterRejectsUnsupportedFormat(tmp_path, toyModel):
    with pytest.raises(AssertionError, match="unsupported format"):
        toyModel.writeFluxConstraints(tmp_path / "bounds.csv", format="json")


def testAtomTransitionRoundTrip(tmp_path, toyModel, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    toyModel.writeAtomTransitions(transitionsFile)

    lines = transitionsFile.read_text().strip().split("\n")
    assert len(lines) == len(toyModel.reactions)
    # b_d is reversible, so it must be written with the reversible arrow
    assert any(line.startswith("b_d\tB <==> D") for line in lines)

    # reading the file back into a model without transitions reproduces them
    toyNetwork.readAtomTransitions(transitionsFile)
    for reaction in toyModel.reactions:
        original = repr(reaction.transitions[0])
        restored = repr(toyNetwork.reactions.get_by_id(reaction.id).transitions[0])
        assert restored == original


def testAtomTransitionReaderRejectsUnknownReaction(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("not_a_reaction\tA --> B\tabc : abc\n")
    with pytest.raises(AssertionError, match="missing from self"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsUnknownReactant(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tF --> B\tabc : abc\n")
    with pytest.raises(AssertionError, match="reactant missing"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsUnknownProduct(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tA --> F\tabc : abc\n")
    with pytest.raises(AssertionError, match="product missing"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsMismatchedReactantIndices(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("bc_de\tB + C --> D\tabc : abc\n")
    with pytest.raises(AssertionError, match="differing number of reactant names"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsMismatchedProductIndices(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("b_ec\tB --> E + C\tabc : abc\n")
    with pytest.raises(AssertionError, match="differing number of product names"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsUnbalancedIndices(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tA --> B\tabc : ab\n")
    with pytest.raises(AssertionError, match="differing number of indices"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsDuplicatedProductIndices(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tA --> B\tabc : aab\n")
    with pytest.raises(AssertionError, match="duplicated indices"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsDuplicatedReactantIndices(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tA --> B\taab : abc\n")
    with pytest.raises(AssertionError, match="duplicated indices"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsDisjointIndexSets(tmp_path, toyNetwork):
    transitionsFile = tmp_path / "transitions.txt"
    transitionsFile.write_text("a_b\tA --> B\tabc : xyz\n")
    with pytest.raises(AssertionError, match="differing number of reactant and product"):
        toyNetwork.readAtomTransitions(transitionsFile)


def testAtomTransitionReaderRejectsUnsupportedFormat(tmp_path, toyNetwork):
    with pytest.raises(AssertionError, match="unsupported format"):
        toyNetwork.readAtomTransitions(tmp_path / "t.txt", format="jqmm1")


def testAtomTransitionWriterRejectsUnsupportedFormat(tmp_path, toyModel):
    with pytest.raises(AssertionError, match="unsupported format"):
        toyModel.writeAtomTransitions(tmp_path / "t.txt", format="jqmm1")
