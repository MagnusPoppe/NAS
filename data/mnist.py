import numpy as np
from tensorflow import keras

# Preparing data:
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val = x_train[50000:] / 255.0
y_val = keras.utils.to_categorical(y_train[50000:], num_classes=10)
x_train = x_train[:50000] / 255.0
y_train = keras.utils.to_categorical(y_train[:50000], num_classes=10)
x_test = x_test / 255.0

x_train = np.reshape(x_train, [50000, 28, 28, 1])
x_val = np.reshape(x_val, [10000, 28, 28, 1])
x_test = np.reshape(x_test, [10000, 28, 28, 1])

def get_training_data() -> (np.ndarray, np.ndarray):
    return x_train, y_train


def get_validation_data() -> (np.ndarray, np.ndarray):
    return x_val, y_val


def get_test_data() -> (np.ndarray, np.ndarray):
    return x_test, y_test


