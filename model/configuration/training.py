"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file fetches the configurations for the model.

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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.download_elements import download_periodic
import csv
import python_ta as pyta
from typing import Union


def atoms() -> dict[str, set]:
    """Returns a dictionary of atom features.

    Returns:
        A dictionary containing the following keys and sets of values:
        - 'element': A set of strings representing the atomic symbols.
        - 'valence_electrons': A set of integers representing the possible valence electron counts.
        - 'hydrogen_bonds': A set of integers representing the possible numbers of hydrogen bonds.
        - 'orbital_hybridization': A set of strings representing the possible orbital hybridizations.
    """
    filename = download_periodic()
    symbols = set()
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            symbols.add(str(row[1]))
    return {
        "element": {
            "B",
            "Br",
            "C",
            "Ca",
            "Cl",
            "F",
            "H",
            "I",
            "N",
            "Na",
            "O",
            "P",
            "S",
        },
        "valence_electrons": {0, 1, 2, 3, 4, 5, 6},
        "hydrogen_bonds": {0, 1, 2, 3, 4},
        "orbital_hybridization": {"s", "sp", "sp2", "sp3"},
    }


def bonds() -> dict[str, set]:
    """Returns a dictionary of bond features.

    Returns:
        A dictionary containing the following keys and sets of values:
        - 'bond_type': A set of strings representing the possible bond types.
        - 'conjugation_state': A set of booleans representing the possible conjugation states.
    """
    return {
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugation_state": {True, False},
    }


def edge_network() -> dict[str, str]:
    """Returns a dictionary of edge network features.

    Returns:
        A dictionary containing the following keys and string values:
        - 'kernel_initializer': A string representing the type of kernel initializer.
        - 'bias_initializer': A string representing the type of bias initializer.
    """
    return {
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
    }


def model() -> dict[str, Union[str, int, list[Union[int, str]], bool, float, None]]:
    """Returns a dictionary of model features.

    Returns:
        A dictionary containing the following keys and values of various types:
        - 'edge_update': A string representing the type of edge update.
        - 'batch_size': An integer representing the batch size.
        - 'message_units': An integer representing the number of message units.
        - 'message_steps': An integer representing the number of message steps.
        - 'num_attention_heads': An integer representing the number of attention heads.
        - 'dense_units': A list of integers representing the units of dense layers.
        - 'activation': A list of strings representing the types of activation functions.
        - 'loss': A string representing the type of loss function.
        - 'optimizer': A string representing the type of optimizer.
        - 'learning_rate': A float representing the learning rate.
        - 'beta_1': A float representing the value of the beta_1 parameter.
        - 'beta_2': A float representing the value of the beta_2 parameter.
        - 'epsilon': A float representing the value of the epsilon parameter.
        - 'weight_decay': Either None or a float representing the weight decay.
        - 'momentum': Either None or a float representing the momentum.
        - 'nesterov': A boolean representing whether or not to use Nesterov momentum.
        - 'clipnorm': Either None or a float representing the maximum norm for gradient clipping.
        - 'clipvalue': Either None or a float representing the maximum absolute value for gradient clipping.
        - 'use_ema': A boolean representing whether or not to use exponential moving average.
        - 'ema_momentum': A float representing the momentum for exponential moving average.
        - 'metrics': A list of strings representing the types of evaluation metrics.
        - 'tensorboard': A boolean representing whether or not to use TensorBoard.
        - 'plot_model': A boolean representing whether or not to plot the model architecture.
        - 'save_model': A boolean representing whether or not to save the trained model.
        - 'epochs': An integer representing the number of training epochs.
        - 'strategy': A string representing the type of strategy for distributed training.
    """
    # At the moment we only support the following few naive customizations.
    return {
        # Either GRU, LSTM, SimpleRNN or StackedRNN
        "edge_update": "GRU",
        "batch_size": 32,
        "message_units": 64,
        "message_steps": 4,
        # Quadratic Complexity with respect to attention heads
        "num_attention_heads": 8,
        # The number of items in dense_units and activatiosn should be same
        "dense_units": [512],
        # Either relu, tanh, sigmoid, softmax, softplus, softsign, hard_sigmoid,
        # hard_tanh, elu, selu, linear
        "activation": ["relu"],
        # Either categorical_crossentropy, binary_crossentropy, mean_squared_error,
        # mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error,
        # squared_hinge, hinge, categorical_hinge, logcosh, kullback_leibler_divergence,
        # poisson, cosine_proximity
        "loss": "binary_crossentropy",
        # Either adam, rmsprop, adagrad, adadelta, adamax, nadam, Lion, AdamW
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-07,
        "weight_decay": None,
        "momentum": None,
        "nesterov": False,
        "clipnorm": None,
        "clipvalue": None,
        "use_ema": False,
        "ema_momentum": 0.99,
        # Either accuracy, binary_accuracy, categorical_accuracy, top_k_categorical_accuracy,
        # sparse_top_k_categorical_accuracy, AUC, loss
        "metrics": ["loss", "AUC"],
        "tensorboard": True,
        "plot_model": True,
        "save_model": True,
        "epochs": 200,
        # Either MirroredStrategy, TPUStrategy, MultiWorkerMirroredStrategy,
        # CentralStorageStrategy, ParameterServerStrategy
        "strategy": "TPUStrategy",
    }


pyta.check_all(
    config={
        "extra-imports": ["csv", "sys", "os", "typing", "python_ta"],
        "allowed-io": ["atoms"],
        "max-line-length": 120,
    },
    output="pyta_output12.txt",
)
