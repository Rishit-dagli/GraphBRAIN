from model.utils.feature_encoder import AtomFeatureEncoder, BondFeatureEncoder
import sys
import os

sys.path.append(".")
from model.configuration.training import atoms, bonds
from model.utils.feature_encoder import AtomFeatureEncoder, BondFeatureEncoder


def atom_feature_encoder():
    return AtomFeatureEncoder(allowed_feature_sets=atoms())


def bond_feature_encoder():
    return BondFeatureEncoder(allowed_feature_sets=bonds())
