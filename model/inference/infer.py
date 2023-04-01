"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for connecting the model to the website.

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
import sys
import os
import pandas as pd
import python_ta as pyta

sys.path.append(".")
from model.utils.conversions import smile_to_graph
from model.dataset.loader import loader


def predict(smile: list, model: tf.keras.Model) -> tf.Tensor:
    """Predicts the permeability of the given SMILES string.

    Arguments:
        smile (list): The SMILES string (represented as a list) to predict the permeability of.
        model (tf.keras.Model): The model to use to make the prediction.

    Returns:
        tf.Tensor: The predicted permeability of the SMILES string as a `tf.Tensor` object.
    """
    smile_data = loader(
        smile_to_graph(smile), pd.Series([0.0]), batch_size=1, shuffle=False
    )
    return tf.squeeze(model.predict(smile_data), axis=1)


pyta.check_all(
    "model/inference/infer.py",
    config={
        "extra-imports": ["tensorflow", "sys", "os", "pandas", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
        "disable": [],
    },
    output="pyta_outputs/pyta_output8.html",
)
