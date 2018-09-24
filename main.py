import random

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL, Dropout
from modules.module import Module
from tensorflow import keras

operators2D = [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

if __name__ == '__main__':
    input = keras.layers.Input(shape=(784, ))
    dense1 = keras.layers.Dense(834)(input)

    x_input = keras.layers.Input(dense1.shape)
    x_dense1 = keras.layers.Dense(834)(x_input)
    x_dense2 = keras.layers.Dense(834)(x_dense1)

    model = keras.models.Model(inputs=[x_input], outputs=[x_dense2])(dense1)