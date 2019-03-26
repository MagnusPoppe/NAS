import random

from src.buildingblocks.ops.convolution import Conv3x3, Conv5x5
from src.buildingblocks.ops.dense import (
    DenseS as DenseSmall,
    DenseM as DenseMedium,
    DenseL as DenseLarge,
    # Dropout,
)
from src.buildingblocks.ops.pooling import MaxPooling2x2, AvgPooling2x2


def random_sample(collection: list) -> object:
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def random_sample_remove(collection: list) -> object:
    # Selecting a random operation and creating an instance of it.
    # Then deletes the sample
    index = random.randint(0, len(collection) - 1)
    elem = collection.pop(index)
    return elem


def randomized_index(li: [], index_size: int = 0):
    import numpy as np
    index_size = len(li) if index_size == 0 else index_size
    draw = np.arange(len(li))
    np.random.shuffle(draw)
    return draw[:index_size]


def generate_votes(weights: [(object, int)]) -> list:
    """
        Converts a list of (<object>, <votes>)
        :param weights: A list of tuples containing (<object>, <votes>)
    """
    votes = []
    for operation, weight in weights:
        votes += [operation] * weight
    return votes


operators2D = [Conv3x3, Conv5x5, MaxPooling2x2, AvgPooling2x2]
operators1D = [DenseSmall, DenseMedium, DenseLarge]
OPERATORS_2D_WEIGHTS = [
    (Conv3x3, 10),
    (Conv5x5, 7),
    (MaxPooling2x2, 3),
    (AvgPooling2x2, 3),
]

OPERATORS_1D_WEIGHTS = [
    (DenseSmall, 3),
    (DenseMedium, 5),
    (DenseLarge, 7),
    # (Dropout, 5),
]

operators2D_votes = generate_votes(OPERATORS_2D_WEIGHTS)
operators1D_votes = generate_votes(OPERATORS_1D_WEIGHTS)
operators_votes = operators1D_votes + operators2D_votes
registered_modules = []
