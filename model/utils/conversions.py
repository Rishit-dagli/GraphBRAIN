"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for converting SMILES strings to graphs.

The application has the following functions:
_smile_to_molecule(smile: str) -> Chem.rdchem.Mol | None: Return the molecule corresponding to the given SMILES string.

_molecule_to_graph(molecule: Chem.rdchem.Mol) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Return the graph corresponding
to the given molecule.

smile_to_graph(smiles: list[str]) -> tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]: Return the graph
corresponding to the given SMILES strings.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import rdkit
from rdkit import Chem
from model.utils.apply_feature_encoder import atom_feature_encoder, bond_feature_encoder
import tensorflow as tf
import einops
import python_ta as pyta


def _smile_to_molecule(smile: str) -> Chem.rdchem.Mol | None:
    """Return the molecule corresponding to the given SMILES string.

    Arguments:
        smile: a SMILES string

    Returns:
        the molecule corresponding to the given SMILES string, None on failure
    """
    return Chem.MolFromSmiles(smile, sanitize=True)


def _molecule_to_graph(molecule: Chem.rdchem.Mol) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return the graph corresponding to the given molecule.

    Args:
        molecule: a molecule

    Returns:
        a tuple of tensors representing the graph corresponding to the given molecule
    """
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


def smile_to_graph(smiles: list[str]) -> tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Return the graph corresponding to the given SMILES strings.

    Arguments:
        smiles: a list of SMILES strings

    Returns:
        a tuple of ragged tensors representing the graph corresponding to the given SMILES string
    """
    atoms = []
    bonds = []
    pairs = []
    # flag = False

    for smile in smiles:
        molecule = _smile_to_molecule(smile)
        atom_features, bond_features, pair_indices = _molecule_to_graph(molecule)

        # if tf.rank(atom_features) == 2 and tf.rank(bond_features) == 2:
        #         atom_features = einops.rearrange(atom_features, 'n d -> d n')
        #         bond_features = einops.rearrange(bond_features, 'n d -> d n')
        #         flag = True

        atoms.append(atom_features.numpy())
        bonds.append(bond_features.numpy())
        pairs.append(pair_indices.numpy())

    # atoms = einops.rearrange(atoms, 'n d l -> n l d')
    # bonds = einops.rearrange(bonds, 'n d l -> n l d')
    # pairs = einops.rearrange(pairs, 'n l d -> n l d')
    # Ragged tensors are used to represent the variable length of the molecules.
    # This is a requirement for the graph neural network, since the only reason
    # we use GNNs in the foirst place is to handle variable length input.
    return (
        tf.ragged.constant(atoms, dtype=tf.float32),
        tf.ragged.constant(bonds, dtype=tf.float32),
        tf.ragged.constant(pairs, dtype=tf.int64),
    )


pyta.check_all(
    config={
        "extra-imports": ["tensorflow", "rdkit", "einops", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output4.txt",
)
