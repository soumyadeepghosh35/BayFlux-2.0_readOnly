"""Structural copying in bayflux.core.reactionNetwork.

ReactionNetwork.__init__ and ReactionNetwork.copy each reimplement COBRApy's
object copying so that reactions become EnhancedReactions. The gene and group
branches of both were never exercised, and they are where a copy is most
likely to lose information.
"""

import cobra
import pytest

import bayflux


@pytest.fixture
def genedCobraModel(toyCobraModel):
    """The toy network with gene reaction rules, so genes exist to copy."""
    toyCobraModel.reactions.get_by_id("a_b").gene_reaction_rule = "geneA and geneB"
    toyCobraModel.reactions.get_by_id("b_ec").gene_reaction_rule = "geneC"
    toyCobraModel.reactions.get_by_id("d_f").gene_reaction_rule = "geneA or geneD"
    return toyCobraModel


def testConstructionCopiesGenes(genedCobraModel):
    network = bayflux.ReactionNetwork(genedCobraModel)

    assert {gene.id for gene in network.genes} == {
        "geneA", "geneB", "geneC", "geneD"
    }
    assert all(gene._model is network for gene in network.genes)
    assert network.reactions.get_by_id("a_b").gene_reaction_rule == "geneA and geneB"


def testCopyCopiesGenes(genedCobraModel):
    network = bayflux.ReactionNetwork(genedCobraModel)
    duplicate = network.copy()

    assert {gene.id for gene in duplicate.genes} == {gene.id for gene in network.genes}
    assert all(gene._model is duplicate for gene in duplicate.genes)
    assert duplicate.genes.get_by_id("geneA") is not network.genes.get_by_id("geneA")


def testConstructionCopiesGroups(toyCobraModel):
    group = cobra.core.Group("glycolysisLike")
    group.add_members([toyCobraModel.reactions.get_by_id("a_b")])
    toyCobraModel.add_groups([group])

    network = bayflux.ReactionNetwork(toyCobraModel)
    assert len(network.groups) == 1
    assert network.groups[0].id == "glycolysisLike"


def testConstructionPreservesNotesAndAnnotation(toyCobraModel):
    toyCobraModel.notes["provenance"] = "Antoniewicz 2006 Fig. 3"
    toyCobraModel.annotation["doi"] = "10.1016/j.ymben.2006.09.001"

    network = bayflux.ReactionNetwork(toyCobraModel)

    assert network.notes["provenance"] == "Antoniewicz 2006 Fig. 3"
    assert network.annotation["doi"] == "10.1016/j.ymben.2006.09.001"
    # notes and annotation are deep copied, so mutating the original is safe
    toyCobraModel.notes["provenance"] = "changed"
    assert network.notes["provenance"] == "Antoniewicz 2006 Fig. 3"


def testCopyPreservesNotesAndAnnotation(toyCobraModel):
    toyCobraModel.notes["provenance"] = "Antoniewicz 2006 Fig. 3"
    network = bayflux.ReactionNetwork(toyCobraModel)

    duplicate = network.copy()
    assert duplicate.notes["provenance"] == "Antoniewicz 2006 Fig. 3"
    network.notes["provenance"] = "changed"
    assert duplicate.notes["provenance"] == "Antoniewicz 2006 Fig. 3"


def testWriteAtomTransitionsSkipsReactionsWithoutTransitions(
    tmp_path, toySampleableModel
):
    # the three boundary reactions carry no atom transitions
    transitionsFile = tmp_path / "transitions.txt"
    toySampleableModel.writeAtomTransitions(transitionsFile)

    lines = transitionsFile.read_text().strip().split("\n")
    assert len(lines) == 5
    assert not any(line.startswith("a_exchange") for line in lines)


def testWriteFluxConstraintsCanExportEveryReaction(tmp_path, toySampleableModel):
    boundsFile = tmp_path / "bounds.csv"
    toySampleableModel.writeFluxConstraints(boundsFile, exchangesOnly=False)

    import pandas as pd
    written = pd.read_csv(boundsFile, index_col=0)
    assert written.shape[0] == len(toySampleableModel.reactions)


def testConstructionRemapsMetaboliteGroupMembers(toyCobraModel):
    group = cobra.core.Group("threeCarbonPool")
    group.add_members([toyCobraModel.metabolites.get_by_id("A"),
                       toyCobraModel.metabolites.get_by_id("B")])
    toyCobraModel.add_groups([group])

    network = bayflux.ReactionNetwork(toyCobraModel)
    members = network.groups.get_by_id("threeCarbonPool").members

    assert {member.id for member in members} == {"A", "B"}
    # the members must be the copies owned by the new model, not the originals
    for member in members:
        assert member is network.metabolites.get_by_id(member.id)


def testConstructionRemapsGeneGroupMembers(genedCobraModel):
    group = cobra.core.Group("upperPathwayGenes")
    group.add_members([genedCobraModel.genes.get_by_id("geneA")])
    genedCobraModel.add_groups([group])

    network = bayflux.ReactionNetwork(genedCobraModel)
    members = network.groups.get_by_id("upperPathwayGenes").members

    assert {member.id for member in members} == {"geneA"}
    for member in members:
        assert member is network.genes.get_by_id(member.id)


def testConstructionRemapsNestedGroupMembers(toyCobraModel):
    inner = cobra.core.Group("inner")
    inner.add_members([toyCobraModel.reactions.get_by_id("a_b")])
    outer = cobra.core.Group("outer")
    outer.add_members([inner])
    toyCobraModel.add_groups([inner, outer])

    network = bayflux.ReactionNetwork(toyCobraModel)
    members = network.groups.get_by_id("outer").members
    assert {member.id for member in members} == {"inner"}
