"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for encoding features of atoms and bonds.

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

import tensorflow as tf
import numpy as np
import python_ta as pyta


class FeatureEncoder:
    """A superclass that encodes features of atoms and bonds.

    Instance Attributes:
        total_features (int): the total number of features
        feature_mappings (dict): a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initializes a FeatureEncoder object.

        Args:
            allowed_feature_sets (dict[str, set]): a dictionary mapping feature names to sets of allowed feature values

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
        """Encodes the features of an atom or bond.

        Args:
            inputs (tf.Tensor): a tensor representing an atom or bond

        Returns:
            np.ndarray: a numpy array representing the encoded features
        """
        output = np.zeros((self.total_features,))
        for feature_name, feature_mapping in self.feature_mappings.items():
            feature_value = getattr(self, feature_name)(inputs)
            if feature_value not in feature_mapping:
                continue
            output[feature_mapping[feature_value]] = 1.0
        return output


class AtomFeatureEncoder(FeatureEncoder):
    """A subclass of FeatureEncoder that encodes features of atoms based on allowed_feature_sets.

    Instance Attributes:
        - total_features (int): the total number of features
        - feature_mappings (dict): a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initializes an AtomFeatureEncoder object.

        Args:
            allowed_feature_sets (dict[str, set]): a dictionary mapping feature names to sets of allowed feature values

        Returns:
            None
        """
        super().__init__(allowed_feature_sets)

    def element(self, atom: tf.Tensor) -> str:
        """Returns the atomic symbol of an atom.

        Args:
            atom (tf.Tensor): a tensor representing an atom

        Returns:
            str: the atomic symbol of an atom
        """
        return atom.GetSymbol()

    def valence_electrons(self, atom: tf.Tensor) -> int:
        """Returns the number of valence electrons of an atom.

        Args:
            atom (tf.Tensor): a tensor representing an atom

        Returns:
            int: the number of valence electrons of an atom
        """
        return atom.GetTotalValence()

    def hydrogen_bonds(self, atom: tf.Tensor) -> int:
        """Returns the number of hydrogen bonds of an atom.

        Args:
            atom (tf.Tensor): a tensor representing an atom

        Returns:
            str: the number of hydrogen bonds of an atom
        """
        return atom.GetTotalNumHs()

    def orbital_hybridization(self, atom: tf.Tensor) -> str:
        """Returns the orbital hybridization of an atom.

        Arguments:
            atom (tf.Tensor): a tensor representing an atom

        Returns:
            str: the orbital hybridization of an atom
        """
        return atom.GetHybridization().name.lower()


class BondFeatureEncoder(FeatureEncoder):
    """A subclass of FeatureEncoder that encodes features of bonds based on the dict allowed_feature_sets.

    Instance Attributes:
        total_features (int): the total number of features
        feature_mappings (dict): a dictionary mapping feature names to dictionaries mapping feature values to indices
    """

    def __init__(self, allowed_feature_sets: dict[str, set]) -> None:
        """Initializes a BondFeatureEncoder object.

        Args:
            allowed_feature_sets (dict[str, set]): a dictionary mapping feature names to sets of allowed feature values

        Returns:
            None
        """
        super().__init__(allowed_feature_sets)
        self.total_features += 1

    def encode(self, bond: tf.Tensor) -> np.ndarray:
        """Encodes the features of a bond.

        Args:
            bond (tf.Tensor): a tensor representing a bond

        Returns:
            np.ndarray: a numpy array representing the encoded features
        """
        output = np.zeros((self.total_features,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond: tf.Tensor) -> str:
        """Returns the bond type of a bond.

        Args:
            bond (tf.Tensor): a tensor representing a bond

        Returns:
            str: the bond type of a bond
        """
        return bond.GetBondType().name.lower()

    def conjugation_state(self, bond: tf.Tensor) -> bool:
        """Return the conjugation state of a bond.

        Args:
            bond (tf.Tensor): a tensor representing a bond

        Returns:
            bool: the conjugation state of a bond
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
