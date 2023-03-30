"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for connecting the model to the website.

The application has the following functions:
predict(smile: list, model: tf.keras.Model) -> float: Predict the permeability of the given SMILES string.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import tensorflow as tf
import sys
import os
import pandas as pd
import python_ta as pyta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.conversions import smile_to_graph
from dataset.loader import loader


def predict(smile: list, model: tf.keras.Model) -> float:
    """Predict the permeability of the given SMILES string.

    Arguments:
        smile: The SMILES string to predict the permeability of.
        model: The model to use to make the prediction.

    Returns:
        The predicted permeability of the SMILES string.
    """
    smile_data = loader(
        smile_to_graph(smile), pd.Series([0.0]), batch_size=1, shuffle=False
    )
    return tf.squeeze(model.predict(smile_data), axis=1)


pyta.check_all(
    config={
        "extra-imports": ["tensorflow", "sys", "os", "pandas", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output2.txt",
)
