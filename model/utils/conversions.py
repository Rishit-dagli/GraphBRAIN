import rdkit
from rdkit import Chem
from .apply_feature_encoder import atom_feature_encoder, bond_feature_encoder
import tensorflow as tf


def _smile_to_molecule(smile: str):
    return Chem.MolFromSmiles(smile, sanitize=True)


def _molecule_to_graph(molecule: Chem.rdchem.Mol):
    atoms = []
    bonds = []
    pairs = []
    atom_encoder = atom_feature_encoder()
    bond_encoder = bond_feature_encoder()

    for atom in molecule.GetAtoms():
        atoms.append(atom_encoder.encode(atom))
        pairs.append([atom.GetIdx(), atom.GetIdx()])
        bonds.append(bond_encoder.encode(None))
        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pairs.append([atom.GetIdx(), neighbor.GetIdx()])
            bonds.append(bond_encoder.encode(bond))

    # This constitutes the major use of graaphs, each molecule is a graph with
    # the atoms as nodes and the bonds as edges.
    return (
        tf.constant(atoms, dtype=tf.float32),
        tf.constant(bonds, dtype=tf.float32),
        tf.constant(pairs, dtype=tf.int64),
    )


def smile_to_graph(smiles):
    atoms = []
    bonds = []
    pairs = []

    for smiles in smiles:
        molecule = _smile_to_molecule(smiles)
        atom_features, bond_features, pair_indices = _molecule_to_graph(molecule)

        atoms.append(atom_features.numpy())
        bonds.append(bond_features.numpy())
        pairs.append(pair_indices.numpy())

    # Ragged tensors are used to represent the variable length of the molecules.
    # This is a requirement for the graph neural network, since the only reason
    # we use GNNs in the foirst place is to handle variable length input.
    return (
        tf.ragged.constant(atoms, dtype=tf.float32),
        tf.ragged.constant(bonds, dtype=tf.float32),
        tf.ragged.constant(pairs, dtype=tf.int64),
    )
