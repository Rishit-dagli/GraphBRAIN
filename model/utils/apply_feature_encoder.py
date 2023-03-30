"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for applying feature encoders.

The application has the following functions:

atom_feature_encoder() -> AtomFeatureEncoder: Return an AtomFeatureEncoder object
.
bond_feature_encoder() -> BondFeatureEncoder: Return a BondFeatureEncoder object.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

from model.utils.feature_encoder import AtomFeatureEncoder, BondFeatureEncoder
import sys
import os
import python_ta as pyta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configuration.training import atoms, bonds


def atom_feature_encoder() -> AtomFeatureEncoder:
    """Return an AtomFeatureEncoder object.

    Arguments:
        None

    Returns:
        an AtomFeatureEncoder object
    """
    return AtomFeatureEncoder(allowed_feature_sets=atoms())


def bond_feature_encoder() -> BondFeatureEncoder:
    """Return a BondFeatureEncoder object.

    Arguments:
        None

    Returns:
        a BondFeatureEncoder object
    """
    return BondFeatureEncoder(allowed_feature_sets=bonds())


pyta.check_all(
    config={
        "extra-imports": ["sys", "os", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output3.txt",
)
