"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for loading data into a tf.data.Dataset.

The application has the following functions:
    - repeatx(x: tf.TypeSpec, num) -> tf.Tensor: Repeat a tensor along a new axis.

    - merged_batch(x_batch, y_batch) -> tuple: Merge the batch dimension with the atom dimension.

    - loader(x: tf.TypeSpec, y: tf.TypeSpec, batch_size: int = 32, shuffle: bool = True,
                autotune: bool = True, prefetech_buffer_size: int = 2,
                shuffle_buffer_size: int = 1000) -> tf.data.Dataset: Load data into a tf.data.Dataset.

    - load_dataset(dataset: str, batch_size: int = 32, shuffle: bool = True,
                    autotune: bool = True, prefetech_buffer_size: int = 2,
                    shuffle_buffer_size: int = 1000) -> tf.data.Dataset: Load a dataset into a
                    tf.data.Dataset.

    - load_smiles(smiles: str, batch_size: int = 32, shuffle: bool = True,
                    autotune: bool = True, prefetech_buffer_size: int = 2,
                    shuffle_buffer_size: int = 1000) -> tf.data.Dataset: Load a SMILES string into a
                    tf.data.Dataset.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import tensorflow as tf
import einops
from model.dataset.download_dataset import download_dataset
import pandas as pd
import numpy as np
import sys
import os
import python_ta as pyta

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.conversions import smile_to_graph


def repeatx(x: tf.TypeSpec, num) -> tf.Tensor:
    """Repeat a tensor along a new axis.
    Arguments:
        x: Tensor to repeat.
        num: Number of times to repeat.

    Returns:
        Repeated tensor.
    """
    letter = "a"
    formula = ""
    ctr = 0
    for _ in tf.range(tf.rank(x)):
        formula += f"{letter}{ctr} "
        ctr += 1
    return einops.repeat(x, f"{formula}-> ({formula} b)", b=num)


def merged_batch(x_batch, y_batch) -> tuple:
    """Merge the batch dimension with the atom dimension.

    Arguments:
        x_batch: Tuple of atom features, bond features, and pair indices.
        y_batch: Labels.

    Returns:
        Tuple of atom features, bond features, pair indices, and molecule indicator.
    """
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()
    molecule_indices = tf.cumsum(tf.ones_like(num_atoms)) - tf.ones_like(num_atoms)
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
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
    """Create a tf.data.Dataset from a tuple of features and labels.

    Arguments:
        x: Tuple of atom features, bond features, and pair indices.
        y: Labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        autotune: Whether to use tf.data.AUTOTUNE.
        prefetech_buffer_size: Prefetch buffer size.
        shuffle_buffer_size: Shuffle buffer size.
        num_parallel_calls: Number of parallel calls.

    Returns:
        tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, (y)))
    if shuffle:
        if autotune:
            dataset = dataset.shuffle(tf.data.AUTOTUNE)
        else:
            dataset = dataset.shuffle(shuffle_buffer_size)
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


def split_data(data, train_size=0.8, val_size=0.15, test_size=0.05) -> tuple:
    """Split the data into train, validation, and test sets.

    Arguments:
        data: Pandas dataframe.
        train_size: Fraction of data to use for training.
        val_size: Fraction of data to use for validation.
        test_size: Fraction of data to use for testing.

    Returns:
        Tuple of train, validation, and test sets.
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
    filename="BBBP.csv", train_size=0.8, val_size=0.15, test_size=0.05
) -> tuple:
    """Load the BBBP dataset.

    Arguments:
        filename: Name of the file containing the dataset.
        train_size: Fraction of data to use for training.
        val_size: Fraction of data to use for validation.
        test_size: Fraction of data to use for testing.

    Returns:
        Tuple of train, validation, and test sets.
    """
    if not os.path.exists(filename):
        raise ValueError("Dataset not found. Please download the dataset first.")
    data = pd.read_csv(filename, usecols=[1, 2, 3])
    return split_data(data, train_size, val_size, test_size)


pyta.check_all(
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
    },
    output="pyta_output9.txt",
)
