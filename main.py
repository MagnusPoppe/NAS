# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow import keras
import network

# # System for creating neural networks dynamically through evolution.
# Requires input: Inputs
# Returns output: output of last layer for use with any classifier.
#
# Spec:
# Should have a structure so that it can work with any ML framework
# without much adjustments. Only one python class should convert a network
# topology into a model for any of the major ML frameworks.


if __name__ == '__main__':
    def fix(data):
        return np.expand_dims(data, axis=0).transpose(1, 3, 2, 0)

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

    # GENERATING NETWORK/MODEL:
    model = network.generate(inputs=x_train[0].shape, outputs=10)

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    model.summary()

    # RUNNING TRAINING:
    training = model.fit(
        x_train,
        keras.utils.to_categorical(y_train, num_classes=10),
        epochs=10,
        batch_size=50,
        validation_data=(
            x_val, keras.utils.to_categorical(y_val, num_classes=10))
    )

    # TEST DATA:
    # x_test = fix(x_test)
    # x_test = x_test.astype('float32')
    # x_test /= 255
    # x_test = np.reshape(x_test, (50000, 784))

    print("Ran ok!")
