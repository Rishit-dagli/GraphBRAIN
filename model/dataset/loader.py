import tensorflow as tf
import einops
from model.dataset.download_dataset import download_dataset
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.conversions import smile_to_graph


def repeatx(x, num):
    letter = "a"
    formula = ""
    ctr = 0
    for _ in tf.range(tf.rank(x)):
        formula += f"{letter}{ctr} "
        ctr += 1
    return einops.repeat(x, f"{formula}-> ({formula} b)", b=num)


def merged_batch(x_batch, y_batch):
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    molecule_indices = tf.cumsum(tf.ones_like(num_atoms)) - tf.ones_like(num_atoms)
    molecule_indicator = repeatx(molecule_indices, num_atoms)

    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1], axis=0)

    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    # Ragged Tensors to Tensor
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def loader(
    x,
    y,
    batch_size=32,
    shuffle=True,
    autotune=True,
    prefetech_buffer_size=2,
    shuffle_buffer_size=1024,
    num_parallel_calls=8,
):
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


def split_data(data, train_size=0.8, val_size=0.15, test_size=0.05):
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


def bbbp_dataset(filename="BBBP.csv", train_size=0.8, val_size=0.15, test_size=0.05):
    if not os.path.exists(filename):
        raise ValueError("Dataset not found. Please download the dataset first.")
    data = pd.read_csv(filename, usecols=[1, 2, 3])
    return split_data(data, train_size, val_size, test_size)
