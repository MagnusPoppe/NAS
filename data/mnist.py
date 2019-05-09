import numpy as np
from tensorflow import keras

# Preparing data:
TRAINING_CASES = 60000
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val = x_train[TRAINING_CASES:] / 255.0
y_val = keras.utils.to_categorical(y_train[TRAINING_CASES:], num_classes=10)
x_train = x_train[:TRAINING_CASES] / 255.0
y_train = keras.utils.to_categorical(y_train[:TRAINING_CASES], num_classes=10)
x_test = x_test / 255.0

x_train = np.reshape(x_train, [len(x_train), 28, 28, 1])
x_test = np.reshape(x_test, [len(x_test), 28, 28, 1])
if len(x_val > 0):
    x_val = np.reshape(x_val, [len(x_val), 28, 28, 1])


def get_training_data(augment=False) -> (np.ndarray, np.ndarray):
    return x_train, y_train


def get_validation_data() -> (np.ndarray, np.ndarray):
    if len(x_val) > 0:
        return x_val, y_val
    else:
        return None, None


def get_test_data() -> (np.ndarray, np.ndarray):
    return x_test, y_test
