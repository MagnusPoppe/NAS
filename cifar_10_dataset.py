import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

from firebase.upload import update_status

def cifar10_configure(classes):  # -> (function, function):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_val = x_train[45000:] / 255
    y_val = y_train[45000:]
    x_train = x_train[:45000] / 255
    y_train = y_train[:45000]
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)


    gpu_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8)
    )
    keras.backend.set_session(tf.Session(config=gpu_config))


    def train(population: list, epochs=1.2, batch_size=64, prefix="--> "):
        print(prefix + "Running training for {} epochs on {} models |".format(epochs, len(population)), end="",
              flush=True)
        started = time.time()
        for i, individ in enumerate(population):
            training_epochs = int(epochs * len(individ.keras_operation.layers)) if epochs > 0 else 1
            with tf.device("/gpu:0"):
                # DEFINING FUNCTIONS FOR COMPILATION
                sgd = keras.optimizers.Adam(lr=0.01)
                loss = keras.losses.categorical_crossentropy

                model = individ.keras_operation

                # RUNNING TRAINING:
                update_status("Training {} for {} epochs ( {}/{} models )"
                              .format(individ.ID, training_epochs, i + 1, len(population)))
                model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
                metrics = model.fit(
                    x_train,
                    y_train,
                    epochs=training_epochs,
                    batch_size=batch_size,
                    verbose=0,
                    validation_data=(x_val, y_val)
                )
                individ.fitness = metrics.history['val_acc'][-1]
                print("=", end="", flush=True)
            individ.epochs_trained += training_epochs
        print("| (elapsed time: {} sec)".format(int(time.time() - started)))

    def evaluate(population: list, compiled=True, prefix="--> "):
        print(prefix + "Evaluating {} models".format(len(population)))
        for individ in population:
            with tf.device("/gpu:0"):
                # DEFINING FUNCTIONS FOR COMPILATION
                sgd = keras.optimizers.Adam(lr=0.01)
                loss = keras.losses.categorical_crossentropy

                model = individ.keras_operation
                if not compiled:
                    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
                metrics = model.evaluate(x_test, y_test, verbose=0)
                individ.fitness = metrics[1]  # Accuracy


    return train, evaluate, "CIFAR 10", (32, 32, 3)


if __name__ == '__main__':
    cifar10_configure(classes=10)
