"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for applying feature encoders.

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

import sys
import os

sys.path.append(".")
from model.configuration.training import atoms, bonds
from model.utils.feature_encoder import AtomFeatureEncoder, BondFeatureEncoder


def atom_feature_encoder() -> AtomFeatureEncoder:
    """Returns an AtomFeatureEncoder object.

    Returns:
        AtomFeatureEncoder: an AtomFeatureEncoder object
    """
    return AtomFeatureEncoder(allowed_feature_sets=atoms())


def bond_feature_encoder() -> BondFeatureEncoder:
    """Returns a BondFeatureEncoder object.

    Returns:
        BondFeatureEncoder: a BondFeatureEncoder object
    """
    return BondFeatureEncoder(allowed_feature_sets=bonds())


if __name__ == "__main__":
    import python_ta as pyta

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyta.check_all(
        os.path.join(path, "model", "utils", "apply_feature_encoder.py"),
        config={
            "extra-imports": [
                "sys",
                "os",
                "python_ta",
                "model.configuration.training",
                "model.utils.feature_encoder",
            ],
            "allowed-io": [],
            "max-line-length": 120,
            "disable": ["E9992", "C0413", "W0611", "E9997"],
        },
        output=os.path.join(path, "pyta_outputs", "apply_feature_encoder.html"),
    )
