import random

from src.MOOA.NSGA_II import nsga_ii
from src.ea_nas import architectural_operators as moo
from src.ea_nas.evolutionary_operations.mutation_for_operators import mutate


def optimize(population, selection, steps, config):
    for i in range(steps):
        # Preparation:
        children = []

        # Mutation:
        for selected in selection(population, config.population_size):
            mutated = mutate(selected)
            if mutated:
                children += [mutated]

        # Training networks:
        children = list(set(children))  # Preventing inbreeding

        # Elitism:
        population += children
        population = nsga_ii(
            population,
            moo.architecture_objectives(),
            moo.architecture_domination_operator(
                moo.architecture_objectives()
            ),
            config,
            force_moo=True
        )

        keep = len(population) - config.population_size
        population, removed = population[keep:], population[:keep]
        avg_size = sum([len(x.children) for x in population]) / len(population)
        print(f"   - Progress: {int(i/steps*100)} %. Average size: {avg_size} ops, best size: {len(population[-1].children)}, worst size: {len(population[0].children)}", end="\r")
    print(f"   - Architectures Optimized! Average size: {avg_size} ops          ")
    return population
