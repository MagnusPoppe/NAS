import random

from src.evolutionary_operations.mutation_for_operators import mutate
from src.evolutionary_operations.mutation_operators import append
from src.helpers import random_sample, operators1D_votes, operators2D_votes
from src.buildingblocks.module import Module

def init_population(individs, in_shape, network_min_layers=1, network_max_layers=10):
    population = []
    for i in range(individs):
        root = Module()
        for i in range(random.randint(network_min_layers, network_max_layers)):
            if i > 0:
                root = mutate(root, make_copy=False)
            else:
                first = random_sample(operators1D_votes) if len(in_shape) == 2 else random_sample(operators2D_votes)
                root = append(root, first())
        population += [root]
    return population
