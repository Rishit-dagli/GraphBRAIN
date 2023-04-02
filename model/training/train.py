"""CSC111 Winter 2023 Final Project: Graph Brain

This module implements a function `create_and_train` that creates a deep learning model and trains it.
The configuration of the model is set in a separate configuration file called `training.py`.
The model is created using the function `create_model` from the module `model.training.build_model`.
The input and output shapes of the model are determined by the shapes of the training data `x_train` and `y_train`.
The training data is loaded using the function `loader` from the module `model.dataset.loader`.

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import datetime

sys.path.append(".")
from model.training.build_model import create_model
from model.configuration.training import edge_network, model, data_splits
from model.dataset.loader import bbbp_dataset, loader
from model.dataset.download_dataset import download_dataset


def create_and_train() -> None:
    """Creates and trains a deep learning model.

    Args:
        None

    Returns:
        None
    """
    model = create_model(
        atom_dim=x_train[0][0][0].shape[0],
        bond_dim=x_train[1][0][0].shape[0],
        batch_size=config["batch_size"],
        message_units=config["message_units"],
        message_steps=config["message_steps"],
        num_attention_heads=config["num_attention_heads"],
        dense_units=config["dense_units"],
        activation=config["activation"],
        edge_update=config["edge_update"],
    )
    if config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
            amsgrad=config["amsgrad"],
            weight_decay=config["weight_decay"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config["learning_rate"],
            rho=config["rho"],
            momentum=config["momentum"],
            epsilon=config["epsilon"],
            centered=config["centered"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=config["learning_rate"],
            initial_accumulator_value=config["initial_accumulator_value"],
            epsilon=config["epsilon"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=config["learning_rate"],
            rho=config["rho"],
            epsilon=config["epsilon"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "adafactor":
        optimizer = tf.keras.optimizers.Adafactor(
            learning_rate=config["learning_rate"],
            initial_accumulator_value=config["initial_accumulator_value"],
            beta_2_decay=config["beta_2_decay"],
            epsilon_1=config["epsilon_1"],
            epsilon_2=config["epsilon_2"],
            clip_threshold=config["clip_threshold"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config["learning_rate"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "nadam":
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "ftrl":
        optimizer = tf.keras.optimizers.Ftrl(
            learning_rate=config["learning_rate"],
            learning_rate_power=config["learning_rate_power"],
            initial_accumulator_value=config["initial_accumulator_value"],
            l1_regularization_strength=config["l1_regularization_strength"],
            l2_regularization_strength=config["l2_regularization_strength"],
            l2_shrinkage_regularization_strength=config[
                "l2_shrinkage_regularization_strength"
            ],
            beta=config["beta"],
            weight_decay=config["weight_decay"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "lion":
        optimizer = tf.keras.optimizers.Lion(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
            weight_decay=config["weight_decay"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "adamax":
        optimizer = tf.keras.optimizers.experimental.Adamax(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )
    elif config["optimizer"] == "adamw":
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            epsilon=config["epsilon"],
            weight_decay=config["weight_decay"],
            clipnorm=config["clipnorm"],
            clipvalue=config["clipvalue"],
            use_ema=config["use_ema"],
            ema_momentum=config["ema_momentum"],
        )

    metrics = []
    for metric in config["metrics"]:
        if metric == "accuracy":
            metrics.append(tf.keras.metrics.Accuracy())
        elif metric == "AUC":
            metrics.append(tf.keras.metrics.AUC())
        elif metric == "kl":
            metrics.append(tf.keras.metrics.KLDivergence())

    model.compile(optimizer=optimizer, loss=config["loss"], metrics=metrics)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = None
    if config["tensorboard"]:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        print("Saving Tensorboard logs at: " + log_dir)

    if tensorboard_callback is not None:
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=config["epochs"],
            verbose=2,
            class_weight={0: 2.0, 1: 0.5},
            callbacks=[tensorboard_callback],
        )
    else:
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=config["epochs"],
            verbose=2,
            class_weight={0: 2.0, 1: 0.5},
        )

    if config["save_model"]:
        tf.saved_model.save(model, log_dir + "/model")

    if config["plot_model"]:
        tf.keras.utils.plot_model(
            model, to_file=log_dir + "/model.png", show_shapes=True
        )

    test_results = model.evaluate(test_dataset, verbose=2)
    print("Test loss:", test_results[0])


if __name__ == "__main__":
    edge_config = edge_network()
    data_config = data_splits()
    config = model()
    download_dataset()
    x_train, y_train, x_valid, y_valid, x_test, y_test = bbbp_dataset(
        train_size=data_config["train"],
        val_size=data_config["validation"],
        test_size=data_config["test"],
    )
    train_dataset = loader(
        x_train,
        y_train,
        batch_size=config["batch_size"],
        shuffle=True,
        autotune=True,
        shuffle_buffer_size=data_config["shuffle_buffer_size"],
    )
    valid_dataset = loader(
        x_valid,
        y_valid,
        batch_size=config["batch_size"],
        shuffle=True,
        autotune=True,
        shuffle_buffer_size=data_config["shuffle_buffer_size"],
    )
    test_dataset = loader(
        x_test,
        y_test,
        batch_size=config["batch_size"],
        shuffle=True,
        autotune=True,
        shuffle_buffer_size=data_config["shuffle_buffer_size"],
    )

    strategy = None
    if config["strategy"]:
        if config["strategy"] == "MirroredStrategy":
            strategy = tf.distribute.MirroredStrategy(
                cluster_resolver=config["cluster_resolver"]
            )
        elif config["strategy"] == "TPUStrategy":
            strategy = tf.distribute.TPUStrategy()
        elif config["strategy"] == "MultiWorkerMirroredStrategy":
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                cluster_resolver=config["cluster_resolver"]
            )
        elif config["strategy"] == "CentralStorageStrategy":
            strategy = tf.distribute.experimental.CentralStorageStrategy(
                cluster_resolver=config["cluster_resolver"]
            )
        elif config["strategy"] == "ParameterServerStrategy":
            strategy = tf.distribute.experimental.ParameterServerStrategy(
                cluster_resolver=config["cluster_resolver"]
            )

    if strategy:
        with strategy.scope():
            create_and_train()
    else:
        create_and_train()
