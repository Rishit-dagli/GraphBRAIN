import rdkit
from rdkit import Chem
from .apply_feature_encoder import atom_feature_encoder, bond_feature_encoder
import tensorflow as tf

def smile_to_molecule(smile: str):
    return Chem.MolFromSmiles(smile, sanitize=True)

def molecule_to_graph(molecule: Chem.rdchem.Mol):
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

    return tf.constant(atoms), tf.constant(bonds), tf.constant(pairs)