import sys
import os

sys.path.append(".")
from model.dataset.download_elements import download_periodic
import csv


def atoms():
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


def bonds():
    return {
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugation_state": {True, False},
    }


def edge_network():
    return {
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
    }


def data_splits():
    return {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
        "shuffle_buffer_size": 1024
    }


def model():
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
        # Either adam, rmsprop, adagrad, adadelta, adafactor, adamax, nadam, lion, adamw, ftrl, sgd
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-07,
        "amsgrad": False,
        "rho": 0.9,
        "centered": False,
        "weight_decay": None,
        "initial_accumulator_value": 0.1,
        "beta_2_decay": -0.8,
        "epsilon_1": 1e-30,
        "epsilon_2": 0.001,
        "clip_threshold": 1.0,
        "momentum": None,
        "nesterov": False,
        "clipnorm": None,
        "clipvalue": None,
        "use_ema": False,
        "ema_momentum": 0.99,
        "learning_rate_power": -0.5,
        "l1_regularization_strength": 0.0,
        "l2_regularization_strength": 0.0,
        "l2_shrinkage_regularization_strength": 0.0,
        "beta": 0.0,
        # Use any combination of AUC, accuracy, loss, kl
        "metrics": ["loss", "AUC"],
        "tensorboard": True,
        "plot_model": True,
        "save_model": True,
        "epochs": 200,
        # Either MirroredStrategy, TPUStrategy, MultiWorkerMirroredStrategy,
        # CentralStorageStrategy, ParameterServerStrategy or None
        "strategy": None,
        "cluster_resolver": None,
    }
