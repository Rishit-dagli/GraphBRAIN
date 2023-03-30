"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for encoding features of atoms and bonds.

The application has the following functions:

FeatureEncoder: A class that encodes features of atoms and bonds.
    encode(self, inputs: tf.Tensor) -> np.ndarray: Encode the features of an atom or bond.

AtomFeatureEncoder: A class that encodes features of atoms.
    element(self, atom: tf.Tensor) -> str: Return the atomic symbol of an atom.

    hydrogen_bonds(self, atom: tf.Tensor) -> int: Return the number of hydrogen bonds of an atom.

    orital_hybridization(self, atom: tf.Tensor) -> str: Return the orbital hybridization of an atom.

    valence_electrons(self, atom: tf.Tensor) -> int: Return the number of valence electrons of an atom.

BondFeatureEncoder: A class that encodes features of bonds.
    bond_type(self, bond: tf.Tensor) -> str: Return the type of a bond.

    conjugation_state(self, bond: tf.Tensor) -> bool: Return the conjugation state of a bond.

    encode(self, bond: tf.Tensor) -> np.ndarray: Encode the features of a bond.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import tensorflow as tf
import numpy as np
import python_ta as pyta


class FeatureEncoder:
    """A class that encodes features of atoms and bonds.

    Instance Attributes:
        - total_features: the total number of features
        - feature_mappings: a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initialize a FeatureEncoder object.

        Arguments:
            allowed_feature_sets: a dictionary mapping feature names to sets of allowed feature values

        Returns:
            None
        """
        self.total_features = 0
        self.feature_mappings = {}
        for feature_name, feature_set in allowed_feature_sets.items():
            sorted_feature_set = sorted(list(feature_set))
            self.feature_mappings[feature_name] = dict(
                zip(
                    sorted_feature_set,
                    range(self.total_features, len(feature_set) + self.total_features),
                )
            )
            self.total_features += len(feature_set)

    def encode(self, inputs: tf.Tensor) -> np.ndarray:
        """Encode the features of an atom or bond.

        Arguments:
            inputs: a tensor representing an atom or bond

        Returns:
            a numpy array representing the encoded features
        """
        output = np.zeros((self.total_features,))
        for feature_name, feature_mapping in self.feature_mappings.items():
            feature_value = getattr(self, feature_name)(inputs)
            if feature_value not in feature_mapping:
                continue
            output[feature_mapping[feature_value]] = 1.0
        return output


class AtomFeatureEncoder(FeatureEncoder):
    """A class that encodes features of atoms.

    Instance Attributes:
        - total_features: the total number of features
        - feature_mappings: a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initialize an AtomFeatureEncoder object.

        Arguments:
            allowed_feature_sets: a dictionary mapping feature names to sets of allowed feature values

        Returns:
            None
        """
        super().__init__(allowed_feature_sets)

    def element(self, atom: tf.Tensor) -> str:
        """Return the atomic symbol of an atom.

        Arguments:
            atom: a tensor representing an atom

        Returns:
            the atomic symbol of an atom
        """
        return atom.GetSymbol()

    def valence_electrons(self, atom: tf.Tensor) -> int:
        """Return the number of valence electrons of an atom.

        Arguments:
            atom: a tensor representing an atom

        Returns:
            the number of valence electrons of an atom
        """
        return atom.GetTotalValence()

    def hydrogen_bonds(self, atom: tf.Tensor) -> int:
        """Return the number of hydrogen bonds of an atom.

        Arguments:
            atom: a tensor representing an atom

        Returns:
            the number of hydrogen bonds of an atom
        """
        return atom.GetTotalNumHs()

    def orbital_hybridization(self, atom: tf.Tensor) -> str:
        """Return the orbital hybridization of an atom.

        Arguments:
            atom: a tensor representing an atom

        Returns:
            the orbital hybridization of an atom
        """
        return atom.GetHybridization().name.lower()


class BondFeatureEncoder(FeatureEncoder):
    """A class that encodes features of bonds.

    Instance Attributes:
        - total_features: the total number of features
        - feature_mappings: a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initialize a BondFeatureEncoder object.

        Arguments:
            allowed_feature_sets: a dictionary mapping feature names to sets of allowed feature values

        Returns:
            None
        """
        super().__init__(allowed_feature_sets)
        self.total_features += 1

    def encode(self, bond: tf.Tensor) -> np.ndarray:
        """Encode the features of a bond.

        Arguments:
            bond: a tensor representing a bond

        Returns:
            a numpy array representing the encoded features
        """
        output = np.zeros((self.total_features,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond: tf.Tensor) -> str:
        """Return the bond type of a bond.

        Arguments:
            bond: a tensor representing a bond

        Returns:
            the bond type of a bond
        """
        return bond.GetBondType().name.lower()

    def conjugation_state(self, bond: tf.Tensor) -> bool:
        """Return the conjugation state of a bond.

        Arguments:
            bond: a tensor representing a bond

        Returns:
            the conjugation state of a bond
        """
        return bond.GetIsConjugated()


pyta.check_all(
    config={
        "extra-imports": ["tensorflow", "numpy", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output5.txt",
)
