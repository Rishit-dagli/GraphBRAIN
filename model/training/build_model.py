"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for building the model.

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

import os
import tensorflow as tf
import einops


class EdgeNetwork(tf.keras.layers.Layer):
    """Edge network for message passing.

    Instance Attributes:
        atom_dim (int): Dimension of atom features
        bond_dim (int): Dimension of bond features
        kernel (tf.Tensor): Kernel for edge network
        bias (tf.Tensor): Bias for edge network
        built (bool): Whether the layer has been built
    """

    def build(
        self,
        input_shape: tf.TensorShape,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        """Builds the layer.

        Args:
            input_shape (tf.TensorShape): Shape of input
            kernel_initializer (str, optional): Initializer for kernel. Defaults to "glorot_uniform".
            bias_initializer (str, optional): Initializer for bias. Defaults to "zeros".

        Returns:
            None
        """
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer=kernel_initializer,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim),
            initializer=bias_initializer,
            name="bias",
        )
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Aggregated features
        """
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        pair_indices_0 = []
        for x in pair_indices:
            pair_indices_0.append(x[0])
        pair_indices_1 = []
        for x in pair_indices:
            pair_indices_1.append(x[1])

        atom_features_neighbors = tf.gather(atom_features, tf.stack(pair_indices_1, 0))
        atom_features_neighbors = einops.rearrange(
            atom_features_neighbors, "... -> ... ()"
        )
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = einops.rearrange(transformed_features, "... () -> ...")
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices_0,
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(tf.keras.layers.Layer):
    """Message passing layer.

    Instance Attributes:
        units (int): Number of units in the layer.
        steps (int): Number of message passing steps.
        edge_update (str): Type of edge update.
    """

    def __init__(
        self, units: int, steps: int = 4, edge_update: str = "GRU", **kwargs
    ) -> None:
        """Initializes the MessagePassing layer.

        Args:
            units (int): Number of units in the layer.
            steps (int): Number of message passing steps.
            edge_update (str): Type of edge update.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.edge_update = edge_update

    def build(self, input_shape: tf.TensorShape) -> None:
        """Builds the MessagePassing layer.

        Args:
            input_shape (tf.TensorShape): Shape of input tensor.

        Returns:
            None
        """
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        if self.edge_update == "GRU":
            self.update_step = tf.keras.layers.GRUCell(self.atom_dim + self.pad_length)
        elif self.edge_update == "LSTM":
            self.update_step = tf.keras.layers.LSTMCell(self.atom_dim + self.pad_length)
        elif self.edge_update == "SimpleRNN":
            self.update_step = tf.keras.layers.SimpleRNNCell(
                self.atom_dim + self.pad_length
            )
        else:
            self.update_step = tf.keras.layers.StackedRNNCells(
                self.atom_dim + self.pad_length
            )
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the MessagePassing layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor
        """
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform message passing
        for i in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated


class TransformerEncoderReadout(tf.keras.layers.Layer):
    """Transformer encoder readout layer. This layer applies multi-head attention and dense projection
    to the input tensor and returns a global average pooling of the output.

    Instance Attributes:
        partition_padding (PartitionPadding): Partition padding layer
        attention (tf.keras.layers.MultiHeadAttention): Multi-head attention layer
        dense_proj (tf.keras.Sequential): Dense projection layer
        layernorm_1 (tf.keras.layers.LayerNormalization): Layer normalization layer
        layernorm_2 (tf.keras.layers.LayerNormalization): Layer normalization layer
        average_pooling (tf.keras.layers.GlobalAveragePooling1D): Global average pooling layer
    """

    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 64,
        dense_dim: list[int] = [512],
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """Initializes the layer.

        Args:
            num_heads (int): Number of attention heads in the multi-head attention layer.
            embed_dim (int): Embedding dimension in the multi-head attention layer.
            dense_dim (list[int]): Dense dimension in the dense projection layer.
            batch_size (int): Batch size.

        Returns:
            None
        """
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(i, activation="relu") for i in dense_dim]
            + [
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.average_pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Transformed tensor.
        """
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = einops.rearrange(padding_mask, "i ... -> i 1 1 ...")
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)


class PartitionPadding(tf.keras.layers.Layer):
    """Partition padding layer. This layer applies padding to atom feature tensors based on a partition function.
    The padding is necessary because the input data can have varying numbers of atoms per molecule.

    Instance Attributes:
        batch_size (int): Batch size
    """

    def __init__(self, batch_size: int, **kwargs) -> None:
        """Initializes the layer.

        Args:
            batch_size (int): Batch size.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Transformed tensor.
        """
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


def create_model(
    atom_dim: int,
    bond_dim: int,
    batch_size: int = 32,
    message_units: int = 64,
    message_steps: int = 4,
    num_attention_heads: int = 8,
    dense_units: list[int] = [512],
    activation: list[str] = ["relu"],
    edge_update: str = "GRU",
) -> tf.keras.Model:
    """Creates a message passing neural network.

    Args:
        atom_dim (int): The dimension of atom features.
        bond_dim (int): The dimension of bond features.
        batch_size (int, optional): The batch size. Defaults to 32.
        message_units (int): The number of units in the message vectors. Defaults to 64.
        message_steps (int): The number of message passing steps. Defaults to 4.
        num_attention_heads (int): The number of attention heads. Defaults to 8.
        dense_units (list[int]): A list of the number of units in each dense layer. Defaults to [512].
        activation (list[str]): A list of the activation functions for each dense layer. Defaults to ["relu"].
        edge_update (str): The type of edge update method. Defaults to "GRU".

    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    atom_features = tf.keras.layers.Input(
        (atom_dim), dtype="float32", name="atom_features"
    )
    bond_features = tf.keras.layers.Input(
        (bond_dim), dtype="float32", name="bond_features"
    )
    pair_indices = tf.keras.layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = tf.keras.layers.Input(
        (), dtype="int32", name="molecule_indicator"
    )

    # Leverage the functional API to create a model with 3 inputs aka the graph
    # and outputs one scalar.
    x = MessagePassing(message_units, message_steps, edge_update)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    for i in range(len(dense_units)):
        x = tf.keras.layers.Dense(dense_units[i], activation=activation[i])(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model


if __name__ == "__main__":
    import python_ta as pyta

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyta.check_all(
        os.path.join(path, "model", "training", "build_model.py"),
        config={
            "extra-imports": ["tensorflow", "einops", "python_ta", "os"],
            "allowed-io": [],
            "max-line-length": 120,
            "disable": [
                "E9972",
                "W0221",
                "W0201",
                "R0902",
                "W0612",
                "W0102",
                "R0913",
                "R0914",
            ],
        },
        output=os.path.join(path, "pyta_outputs", "build_model.html"),
    )
