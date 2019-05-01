import copy
import time
import random

from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.ea_nas.evolutionary_operations.initialization import init_population
from src.ea_nas.evolutionary_operations.mutation_for_operators import mutate
from src.ea_nas.evolutionary_operations.selection import tournament
from firebase.upload import update_status, upload_population
from src.ea_nas.finalize import try_finish
from src.output import generation_finished
from src.ea_nas import operators as moo
from src.MOOA.NSGA_II import nsga_ii, weighted_overfit_score
import src.jobs.jobs as workers
from src.jobs import garbage_collector

import builtins

builtins.generation = 0


def evolve_architecture(selection: callable, config: Configuration, population: [Module] = None):
    update_status("Creating initial population")

    # initializing population
    if not population:
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

    # Running EA algorithm:
    for generation in range(config.generations):
        config.generation = generation

        # Preparation:
        print("\nGeneration", generation)
        builtins.generation = generation
        children = []

        # Mutation:
        print("--> Mutations:")
        update_status("Mutating")
        for selected in selection(population, config.population_size):
            draw = random.uniform(0, 1)
            mutated = None
            if draw < 0.9:
                print("    - Operation Mutation for {}".format(selected.ID))
                mutated = mutate(selected)
            else:  # elif draw < 0.9:
                print("    - Creating new net randomly")
                mutated = init_population(1, config.input_format, 3, 30)[0]

            if mutated:
                children += [mutated]

        # Training networks:
        children = list(set(children))  # Preventing inbreeding

        # Elitism:
        population += children
        population = workers.start(population, config)
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

        # Removing unused models:
        # if not config.save_all_results:
        #     garbage_collector.collect_garbage(removed, population, config)
        upload_population(population)
        generation_finished(population, config, f"--> Generation {generation} Leaderboards:")
        generation_finished(removed, config, "--> The following individs were removed by elitism:")
        config.results.store_generation(population, generation)

        # Checking for a satisfactory solution
        if any(ind.val_acc() > config.training.acceptable_scores - 0.10 for ind in population):
            population, solved = try_finish(population, config, moo)
            if solved:
                return population
    return population


def run(config):
    print("\n\nEvolving architecture")
    start_time = time.time()
    population = None
    if config.pretrain_dataset:
        print(f"\n\nPre training stage, training on easier dataset {config.pretrain_dataset.dataset_name}")

        # Setting config correctly:
        config.dataset_name = config.pretrain_dataset.dataset_name
        config.dataset_file_name = config.pretrain_dataset.dataset_file_name
        config.dataset_file_path = config.pretrain_dataset.dataset_file_path
        config.training.acceptable_scores = config.pretrain_dataset.accepted_accuracy
        config.input_format = config.pretrain_dataset.input

        # Running pretrain stage:
        population = evolve_architecture(selection=tournament, config=config)

    # Setting config correctly:
    config.dataset_name = config.target_dataset.dataset_name
    config.dataset_file_name = config.target_dataset.dataset_file_name
    config.dataset_file_path = config.target_dataset.dataset_file_path
    config.training.acceptable_scores = config.target_dataset.accepted_accuracy
    config.input_format = config.target_dataset.input

    # Running main training stage:
    _ = evolve_architecture(selection=tournament, config=config, population=population)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
