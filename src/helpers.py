import random

from src.buildingblocks.ops.convolution import Conv3x3, Conv5x5
from src.buildingblocks.ops.dense import DenseS, DenseM, DenseL, Dropout
from src.buildingblocks.ops.pooling import MaxPooling2x2, AvgPooling2x2

operators2D = [Conv3x3, Conv5x5, MaxPooling2x2,
               AvgPooling2x2, Dropout, Dropout]
operators1D = [DenseS, DenseM, DenseL, Dropout, Dropout]
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
