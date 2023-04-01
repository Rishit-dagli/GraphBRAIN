"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for loading data into a tf.data.Dataset.

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
import einops
from model.dataset.download_dataset import download_dataset
import pandas as pd
import numpy as np
import sys
import os
import python_ta as pyta

sys.path.append(".")
from model.utils.conversions import smile_to_graph
from model.dataset.download_dataset import download_dataset


def repeatx(x: tf.Tensor, num: int) -> tf.Tensor:
    """Repeats a tensor along a new axis.

    Args:
        x (tf.TypeSpec): Tensor to repeat.
        num (int): Number of times to repeat.

    Returns:
        tf.Tensor: Repeated tensor.
    """
    letter = "a"
    formula = ""
    ctr = 0
    for _ in tf.range(tf.rank(x)):
        formula += f"{letter}{ctr} "
        ctr += 1
    return einops.repeat(x, f"{formula}-> ({formula} b)", b=num)


def merged_batch(x_batch: tuple, y_batch: tf.Tensor) -> tuple:
    """Merges the batch dimension with the atom dimension.

    Args:
        x_batch (tuple): Tuple of atom features, bond features, and pair indices.
        y_batch (tf.Tensor): Labels.

    Returns:
        tuple: Tuple of atom features, bond features, pair indices, and molecule indicator.
    """
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()
    molecule_indices = tf.cumsum(tf.ones_like(num_atoms)) - tf.ones_like(num_atoms)
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def loader(
    x: tf.TypeSpec,
    y: tf.TypeSpec,
    batch_size: int = 32,
    shuffle: bool = True,
    autotune: bool = True,
    prefetech_buffer_size: int = 2,
    shuffle_buffer_size: int = 1024,
    num_parallel_calls: int = 8,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset from a tuple of features and labels.

    Args:
        x (tf.TypeSpec): Tuple of atom features, bond features, and pair indices.
        y (tf.TypeSpec): Labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        autotune (bool): Whether to use tf.data.AUTOTUNE.
        prefetech_buffer_size (int): Prefetch buffer size.
        shuffle_buffer_size (int): Shuffle buffer size.
        num_parallel_calls (int): Number of parallel calls.

    Returns:
        tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, (y)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if autotune:
        return (
            dataset.batch(batch_size)
            .map(merged_batch, tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
    return (
        dataset.batch(batch_size)
        .map(merged_batch, num_parallel_calls)
        .prefetch(prefetech_buffer_size)
    )


def split_data(
    data: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.15,
    test_size: float = 0.05,
) -> tuple:
    """Splits the data into train, validation, and test sets.

    Args:
        data (pd.DataFrame): Pandas dataframe.
        train_size (float): Fraction of data to use for training.
        val_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.

    Returns:
        tuple: Tuple of train, validation, and test sets.
    """
    permuted_indices = np.random.permutation(np.arange(data.shape[0]))
    train_index = permuted_indices[: int(data.shape[0] * train_size)]
    x_train = smile_to_graph(data.iloc[train_index].smiles)
    y_train = data.iloc[train_index].p_np

    valid_index = permuted_indices[
        int(data.shape[0] * train_size) : int(data.shape[0] * (1.0 - test_size))
    ]
    x_valid = smile_to_graph(data.iloc[valid_index].smiles)
    y_valid = data.iloc[valid_index].p_np

    test_index = permuted_indices[int(data.shape[0] * (1.0 - test_size)) :]
    x_test = smile_to_graph(data.iloc[test_index].smiles)
    y_test = data.iloc[test_index].p_np
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def bbbp_dataset(
    filename: str = "BBBP.csv",
    train_size: float = 0.8,
    val_size: float = 0.15,
    test_size: float = 0.05,
) -> tuple:
    """Loads the BBBP dataset.

    Args:
        filename (str): Name of the file containing the dataset.
        train_size (float): Fraction of data to use for training.
        val_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.

    Returns:
        tuple: Tuple of train, validation, and test sets.
    """
    if not os.path.exists(filename):
        raise ValueError("Dataset not found. Please download the dataset first.")
    data = pd.read_csv(filename, usecols=[1, 2, 3])
    return split_data(data, train_size, val_size, test_size)


pyta.check_all(
    "model/dataset/loader.py",
    config={
        "extra-imports": [
            "tensorflow",
            "einops",
            "pandas",
            "numpy",
            "sys",
            "os",
            "python_ta",
        ],
        "allowed-io": [],
        "max-line-length": 120,
        "disable": [],
    },
    output="pyta_outputs/pyta_output9.html",
)
