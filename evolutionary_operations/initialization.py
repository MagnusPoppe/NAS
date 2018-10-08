import random

from evolutionary_operations.mutation import mutate
from frameworks.keras_decoder import assemble
from modules.module import Module

def init_population(individs, in_shape, classes):
    population = []
    for i in range(individs):
        root = Module()
        for _ in range(random.randint(1, 5)):
            root = mutate(root, in_shape, classes, compilation=False)
        root.keras_tensor = assemble(root, in_shape, classes)
        population += [root]
    return population
