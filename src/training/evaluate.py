import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

from src.output import no_stdout


def evaluate(model, data, labels, device: str, compiled=True):
    with tf.device(device.device):
        # EVALUATING
        predictions = model.predict(data)
        pred = np.argmax(predictions, axis=1)
    return classification_report(labels, pred, output_dict=True)
