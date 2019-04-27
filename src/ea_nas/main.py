import copy
import time
import random

from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.ea_nas.evolutionary_operations.initialization import init_population
from src.ea_nas.evolutionary_operations.mutation_for_operators import mutate
from src.ea_nas.evolutionary_operations.selection import tournament
from firebase.upload import update_status, upload_population
from src.output import generation_finished
from src.ea_nas import operators as moo
from src.MOOA.NSGA_II import nsga_ii, weighted_overfit_score
import src.jobs.jobs as workers
from src.jobs import garbage_collector

import builtins

builtins.generation = 0


def evolve_architecture(selection, config: Configuration):
    update_status("Creating initial population")

    # initializing population
    population = init_population(
        individs=config.population_size,
        in_shape=config.input_format,
        network_min_layers=config.min_size,
        network_max_layers=config.max_size
    )

    # Training initial population:
    population = workers.start(population, config)
    population.sort(key=weighted_overfit_score(config), reverse=True)
    upload_population(population)
    generation_finished(population, config, f"--> Initialization complete. Leaderboards:")
    best = population[0]

    # Running EA algorithm:
    for generation in range(config.generations):
        config.generation = generation

        # Preparation:
        print("\nGeneration", generation)
        builtins.generation = generation
        offsprings = []

        # Mutation:
        print("--> Mutations:")
        update_status("Mutating")
        for i in range(config.population_size):
            draw = random.uniform(0, 1)
            mutated = mutate(best) if draw < 0.9 else init_population(1, config.input_format, 3, 30)[0]
            if mutated:
                offsprings += [mutated]

        # Training networks:
        offsprings = list(set(offsprings))  # Preventing inbreeding

        # Elitism:
        population = [best] + offsprings
        population = workers.start(population, config)
        population.sort(key=weighted_overfit_score(config), reverse=True)

        keep = 1
        best, removed = population[-1], population[:-1]

        upload_population(population)
        generation_finished([best], config, f"--> Generation {generation} Leaderboards:")
        generation_finished(removed, config, "--> The following individs were removed by elitism:")
        config.results.store_generation(population, generation)

        if any(ind.validation_fitness[-1] > config.training.acceptable_scores - 0.10 for ind in population):
            trained, solved = try_finish(population, config)
            if solved:
                config.results.store_generation(trained, generation+1)
                return
            population = [best] + trained
            population = nsga_ii(
                population,
                moo.classification_objectives(config),
                moo.classification_domination_operator(
                    moo.classification_objectives(config)
                ),
                config
            )
            keep = len(population) - config.population_size
            population, removed = population[keep:], population[:keep]


def try_finish(population: [Module], config: Configuration) -> [Module]:
    print(f"--> Possible final solution discovered. Checking...")
    solved = False

    # Changing settings of training steps:
    original_training_settings = copy.deepcopy(config.training)
    config.training.use_restart = False
    config.training.fixed_epochs = True
    config.training.epochs = 1

    # Finding the best networks:
    best = population[:config.compute_capacity(maximum=False)]

    # Performing training step:
    best = workers.start(best, config)

    # Reset settings and return:
    config.training = original_training_settings

    best.sort(key=weighted_overfit_score(config), reverse=True)
    if any(ind.validation_fitness[-1] >= config.training.acceptable_scores for ind in best):
        generation_finished(best, config, "--> Found final solution:")
        solved = True

    return best, solved


def run(config):
    print("\n\nEvolving architecture")
    start_time = time.time()

    evolve_architecture(selection=tournament, config=config)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
