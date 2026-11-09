"""Shared pytest setup and fixtures for the BayFlux test suite.

Importing bayflux first installs the numpy compatibility shim that
bayflux/__init__.py applies (np.object = object). The pinned COBRApy fork
still does `from numpy import object`, so any module that imports cobra
before bayflux fails to import under numpy >= 1.24.
"""

import bayflux  # noqa: F401

import shutil
from collections import defaultdict
from pathlib import Path

import cobra
import pytest
import yaml


# setup.cfg sets python_files = tests/*.py, which matches every .py file
# anywhere under tests/, not just those named test_*.py. Input data such as
# tests/fixtures/toyModel/substrateLabelingDict.py is therefore collected as a
# test module and fails to import. Keep the fixtures directory out of
# collection; conftest.py itself is exempt from python_files.
collect_ignore_glob = ["fixtures/*"]


# Reaction stoichiometry of Fig. 3 of Antoniewicz 2006: six metabolites A to F
# and five reactions. The same network the shipped toy model example uses, and
# small enough that EMU compilation and simulation are effectively free.
TOY_METABOLITES = {
    "A": "C3",
    "B": "C3",
    "C": "C2",
    "D": "C3",
    "E": "C1",
    "F": "C3",
}

TOY_REACTIONS = {
    "a_b": ({"A": -1.0, "B": 1.0}, 0, 500),
    "b_ec": ({"B": -1.0, "E": 1.0, "C": 1.0}, 0, 500),
    "bc_de": ({"B": -1.0, "C": -1.0, "D": 1.0, "E": 2.0}, 0, 500),
    "d_f": ({"D": -1.0, "F": 1.0}, 0, 500),
    "b_d": ({"B": -1.0, "D": 1.0}, -500, 500),
}

TOY_TRANSITIONS = {
    "a_b": ((("A", [1, 2, 3]),), (("B", [1, 2, 3]),)),
    "b_ec": ((("B", [1, 2, 3]),), (("E", [1]), ("C", [2, 3]))),
    "bc_de": (
        (("B", [1, 2, 3]), ("C", [4, 5])),
        (("E", [1]), ("D", [2, 3, 4]), ("E", [5])),
    ),
    "d_f": ((("D", [1, 2, 3]),), (("F", [1, 2, 3]),)),
    "b_d": ((("B", [1, 2, 3]),), (("D", [1, 2, 3]),)),
}

TOY_FLUX_DICT = {
    "a_b": [100, 0],
    "b_d": [110, 50],
    "b_ec": [20, 0],
    "bc_de": [20, 0],
    "d_f": [80, 0],
}

# Labeling of metabolite F predicted by the fluxes above, from Antoniewicz 2006
TOY_F_LABELING = [0.0001, 0.8008, 0.1983, 0.0009]


def buildToyCobraModel():
    """Return the Fig. 3 network as a plain cobra.Model, without transitions."""
    model = cobra.Model("fig3")
    metabolites = {
        name: cobra.Metabolite(name, formula=formula, name=name, compartment="c")
        for name, formula in TOY_METABOLITES.items()
    }
    reactions = []
    for reactionId, (stoichiometry, lowerBound, upperBound) in TOY_REACTIONS.items():
        reaction = cobra.Reaction(reactionId)
        reaction.lower_bound = lowerBound
        reaction.upper_bound = upperBound
        reaction.add_metabolites(
            {metabolites[name]: coefficient
             for name, coefficient in stoichiometry.items()}
        )
        reactions.append(reaction)
    model.add_reactions(reactions)
    return model


def applyToyTransitions(model):
    """Attach the Fig. 3 atom transitions to a bayflux.ReactionNetwork."""
    metabolites = {metabolite.id: metabolite for metabolite in model.metabolites}
    for reactionId, (reactants, products) in TOY_TRANSITIONS.items():
        model.reactions.get_by_id(reactionId).transitions = [
            bayflux.AtomTransition(
                tuple((metabolites[name], indices) for name, indices in reactants),
                tuple((metabolites[name], indices) for name, indices in products),
            )
        ]
    return model


@pytest.fixture
def toyCobraModel():
    """The Fig. 3 network as a plain cobra.Model, with no atom transitions."""
    return buildToyCobraModel()


@pytest.fixture
def toyNetwork():
    """The Fig. 3 network as a bayflux.ReactionNetwork, with no transitions."""
    return bayflux.ReactionNetwork(buildToyCobraModel())


@pytest.fixture
def toyModel():
    """The Fig. 3 network as a bayflux.ReactionNetwork with atom transitions."""
    return applyToyTransitions(bayflux.ReactionNetwork(buildToyCobraModel()))


@pytest.fixture
def toyFluxes(toyModel):
    """A feasible flux vector for the Fig. 3 network."""
    return bayflux.FluxVector(toyModel, TOY_FLUX_DICT)


@pytest.fixture
def toySubstrateLabeling(toyModel):
    """Substrate labeling of A: singly labeled on the second carbon."""
    return {toyModel.metabolites.get_by_id("A"): ((1.0, [0, 1, 0]),)}


@pytest.fixture
def toyOutputEmu(toyModel):
    """The full size EMU of metabolite F, the simulation target."""
    return bayflux.EMU(toyModel.metabolites.get_by_id("F"), [1, 2, 3])


@pytest.fixture
def toyCompiledData(toyModel, toySubstrateLabeling, toyOutputEmu):
    """EMU network compiled for the F EMU of the Fig. 3 network."""
    return bayflux.emuCompile([toyOutputEmu], toyModel, toySubstrateLabeling)


@pytest.fixture
def toyMassDistribution(toyModel, toyOutputEmu):
    """Experimental MDV for F that matches the simulation of TOY_FLUX_DICT."""
    data = defaultdict(list)
    data[toyOutputEmu].append(list(TOY_F_LABELING))
    return bayflux.MassDistribution(toyModel, data)


TOY_EXCHANGES = {
    "a_exchange": ("A", -100.0, -100.0),
    "e_exchange": ("E", 0.0, 500.0),
    "f_exchange": ("F", 0.0, 500.0),
}


def addToyExchanges(model):
    """Add the boundary reactions that make the Fig. 3 network solvable.

    Without them every internal flux is pinned to zero at steady state, and
    warmup point generation has nothing to explore.
    """
    exchanges = []
    for reactionId, (metaboliteId, lowerBound, upperBound) in TOY_EXCHANGES.items():
        reaction = cobra.Reaction(reactionId)
        reaction.lower_bound = lowerBound
        reaction.upper_bound = upperBound
        reaction.add_metabolites({model.metabolites.get_by_id(metaboliteId): -1.0})
        exchanges.append(reaction)
    model.add_reactions(exchanges)
    return model


@pytest.fixture
def toySampleableModel():
    """The Fig. 3 network with transitions and boundary reactions.

    This is the model the shipped command line example uses, so it is the one
    to sample from.
    """
    model = bayflux.ReactionNetwork(addToyExchanges(buildToyCobraModel()))
    return applyToyTransitions(model)


@pytest.fixture
def toySampleableCobraModel():
    """The Fig. 3 network with boundary reactions, as a plain cobra.Model.

    Kept separate from toySampleableModel because cobra's SBML writer expects
    plain cobra objects, not EnhancedReactions.
    """
    return addToyExchanges(buildToyCobraModel())


# Complete input set for the command line executables, in
# tests/fixtures/toyModel: an SBML copy of the network above plus its atom
# transitions, flux bounds, mass distribution data and substrate labeling. The
# SBML file is checked in rather than written by cobra during the test, both to
# keep the end to end tests fast and to keep libsbml's writer, which has
# segfaulted under some coverage configurations, off the test path.
TOY_INPUT_DIRECTORY = Path(__file__).parent / "fixtures" / "toyModel"

TOY_INPUT_FILES = {
    "modelFile": "fig3toyModel.xml",
    "fluxBoundsFile": "fluxBounds.csv",
    "transitionsFile": "transitions.txt",
    "mdvFile": "massDistribution.tab",
    "substrateLabelingDict": "substrateLabelingDict.py",
}

BASE_RUN_CONFIG = {
    "seed": 4242,
    "thinning": 1,
    "centeringSamplesPerTask": 5,
    "bayesianSamplesPerTask": 5,
    "numProposals": 2,
    "subsamplesPerIteration": 3,
    "averageError": 0.0076875,
    "maxFlux": 150.0,
}


@pytest.fixture
def toyRunDirectory(tmp_path, monkeypatch):
    """Stage a runnable input set and return a config writer.

    Returns a callable that takes configuration overrides, writes config.yaml
    into the staged directory and returns that directory. The working directory
    is switched to it for the duration of the test, because both executables
    resolve their input paths and scan for resumable output relative to the
    working directory.
    """
    for name in TOY_INPUT_FILES.values():
        shutil.copy(TOY_INPUT_DIRECTORY / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)

    def writeConfig(**overrides):
        config = dict(BASE_RUN_CONFIG)
        config.update(TOY_INPUT_FILES)
        config.update(overrides)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
        return tmp_path

    return writeConfig
