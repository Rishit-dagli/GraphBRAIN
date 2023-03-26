import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.download_elements import download_periodic
import csv


def atoms():
    filename = download_periodic()
    symbols = set()
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            symbols.add(str(row[1]))
    return {
        "element": symbols,
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
    }