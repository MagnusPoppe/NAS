import time

import tensorflow as tf
from tensorflow import keras
import numpy as np


def mnist_configure(classes): # -> (function, function):
    def fix(data):
        return np.reshape(data, (len(data), 784))

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # VALIDATION DATA:
    x_val = fix(x_train[50000:])
    y_val = y_train[50000:]
    x_val = x_val.astype('float32')
    x_val /= 255

    # TRAINING DATA:
    x_train = fix(x_train[:50000])
    y_train = y_train[:50000]
    x_train = x_train.astype('float32')
    x_train /= 255

    # TEST DATA:
    x_test = fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    # Converting to one-hot targets:
    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy

    def train(population: list, epochs=5, batch_size=64):
        print("--> Running training for {} epochs on {} models ".format(epochs, len(population)), end="", flush=True)
        started = time.time()
        for individ in population:
            model = individ.keras_operation

            # RUNNING TRAINING:
            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
            metrics = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(x_val, y_val)
            )
            individ.fitness = metrics.history['val_acc'][-1]
        print("(elapsed time: {})".format(time.time()-started))

    def evaluate(population: list):
        print("--> Evaluating {} models".format(len(population)))
        for individ in population:
            model = individ.keras_operation
            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
            metrics = model.evaluate(x_test, y_test, verbose=0)
            individ.fitness = metrics[1]  # Accuracy

    return train, evaluate