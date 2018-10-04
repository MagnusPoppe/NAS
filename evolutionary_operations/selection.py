import random

import numpy as np

from evolutionary_operations.mutation import random_mutation
from module_decoder import assemble
from modules.module import Module


def init_population(individs=10, compile_args=((784,), 10)):
    population = []
    for i in range(individs):
        root = Module()
        for _ in range(random.randint(1, 5)):
            root = random_mutation(root, compilation=False, make_copy=False)
        root.keras_tensor = assemble(root, *compile_args)
        population += [root]
    return population


def tournament(population, size):
    individs = np.array(range(len(population)))
    np.random.shuffle(individs)
    individs = individs[:size]
    for i, j in zip(individs[:int(len(individs) / 2)], individs[int(len(individs) / 2):]):
        yield population[i] if (population[i].fitness > population[j].fitness) else population[j]