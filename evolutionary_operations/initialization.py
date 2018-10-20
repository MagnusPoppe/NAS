import random

from evolutionary_operations.mutation_for_operators import mutate
from evolutionary_operations.mutation_operators import append
from frameworks.keras_decoder import assemble
from helpers import random_sample, operators1D, operators2D
from modules.module import Module

def init_population(individs, in_shape, classes, network_min_layers=1, network_max_layers=10):
    population = []
    for i in range(individs):
        root = Module()
        for i in range(random.randint(network_min_layers, network_max_layers)):
            if i > 0:
                root = mutate(root, in_shape, classes, compilation=False)
            else:
                first = random_sample(operators1D) if len(in_shape) == 2 else random_sample(operators2D)
                root = append(root, first())
        root.keras_tensor = assemble(root, in_shape, classes)
        population += [root]
    return population
