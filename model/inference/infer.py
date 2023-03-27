import tensorflow as tf
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.conversions import smile_to_graph
from dataset.loader import loader


def predict(smile: list, model: tf.keras.Model) -> float:
    smile_data = loader(
        smile_to_graph(smile), pd.Series([0.0]), batch_size=1, shuffle=False
    )
    return tf.squeeze(model.predict(smile_data), axis=1)
