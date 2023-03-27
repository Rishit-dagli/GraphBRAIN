import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.conversions import smile_to_graph
from dataset.loader import loader
def inference(smile: str, model: tf.keras.Model) -> str:
    smile_data = loader(smile_to_graph(smile), 0., batch_size=1, shuffle=False)
    return tf.squeeze(model.predict(smile_data), axis=1)