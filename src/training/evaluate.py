import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

from src.output import no_stdout


def evaluate(model, data, labels, device: str, compiled=True, learning_rate=0.0001):
    with tf.device(device):
        # EVALUATING
        if not compiled:
            optimizer = keras.optimizers.Adam(lr=learning_rate)
            loss = keras.losses.categorical_crossentropy
            model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        predictions = model.predict(data)
        pred = np.argmax(predictions, axis=1)
    return classification_report(labels, pred, output_dict=True)
