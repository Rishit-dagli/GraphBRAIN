"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for converting SMILES strings to graphs.

Copyright and Usage Information
===============================
Copyright 2023 Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import rdkit
from rdkit import Chem
from model.utils.apply_feature_encoder import atom_feature_encoder, bond_feature_encoder
import tensorflow as tf
import einops
import python_ta as pyta


def _smile_to_molecule(smile: str) -> Chem.rdchem.Mol:
    """Returns the molecule corresponding to the given SMILES string.

    Args:
        smile (str): a SMILES string

    Returns:
        Chem.rdchem.Mol: the molecule corresponding to the given SMILES string
    """
    molecule = Chem.MolFromSmiles(smile, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def _molecule_to_graph(
    molecule: Chem.rdchem.Mol,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns the graph corresponding to the given molecule.

    Args:
        molecule (Chem.rdchem.Mol): a molecule

    Returns:
        tuple: a tuple of tensors representing the graph corresponding to the given molecule
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


def smile_to_graph(
    smiles: list[str],
) -> tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    """Returns the graph corresponding to the given SMILES strings.

    Args:
        smiles lst[str]: a list of SMILES strings

    Returns:
        tuple: a tuple of ragged tensors representing the graph corresponding to the given SMILES string
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


# pyta.check_all(
#     "model/utils/conversions.py",
#     config={
#         "extra-imports": ["tensorflow", "rdkit", "einops", "python_ta"],
#         "allowed-io": [],
#         "max-line-length": 120,
#         "disable": [],
#     },
#     output="pyta_outputs/pyta_output4.html",
# )
