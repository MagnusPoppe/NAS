import random

from evolutionary_operations.mutation import mutate
from frameworks.keras_decoder import assemble
from modules.module import Module

def init_population(individs, in_shape, classes, network_min_layers=1, network_max_layers=10):
    population = []
    for i in range(individs):
        root = Module()
        for _ in range(random.randint(network_min_layers, network_max_layers)):
            root = mutate(root, in_shape, classes, compilation=False)
        root.keras_tensor = assemble(root, in_shape, classes)
        population += [root]
    return population
