import random
import time

from tensorflow import keras

from modules.convolution import Conv3x3, Conv5x5
from modules.dense import DenseS, DenseM, DenseL, Dropout

operators2D = [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]
operators = operators1D + operators2D
registered_modules = []

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def random_sample_remove(collection):
    # Selecting a random operation and creating an instance of it.
    # Then deletes the sample
    index = random.randint(0, len(collection) - 1)
    elem = collection.pop(index)
    return elem