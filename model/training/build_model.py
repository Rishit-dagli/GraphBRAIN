import tensorflow as tf
import einops


class EdgeNetwork(tf.keras.layers.Layer):
    def build(
        self, input_shape, kernel_initializer="glorot_uniform", bias_initializer="zeros"
    ):
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

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = einops.rearrange(
            atom_features_neighbors, "... -> ... ()"
        )
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = einops.rearrange(transformed_features, "... () -> ...")
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(tf.keras.layers.Layer):
    def __init__(self, units, steps=4, edge_update="GRU", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.edge_update = edge_update

    def build(self, input_shape):
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

    def call(self, inputs):
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
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=[512], batch_size=32, **kwargs
    ):
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

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)


class PartitionPadding(tf.keras.layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
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
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=[512],
    activation=["relu"],
    edge_update="GRU",
):
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
