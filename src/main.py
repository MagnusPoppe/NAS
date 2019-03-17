import time
import random

from src.configuration import Configuration
from src.evolutionary_operations.initialization import init_population
from src.evolutionary_operations.mutation_for_operators import mutate
from src.evolutionary_operations.selection import tournament
from firebase.upload import update_status, upload_population
from src.output import generation_finished
from src.MOOA import operators as moo
from src.MOOA.NSGA_II import nsga_ii
# import src.jobs.TF_launcher as launcher
import src.jobs.job_initializer as workers
from src.jobs import garbage_collector

import builtins

builtins.generation = 0


def evolve_architecture(selection, config: Configuration):
    update_status("Creating initial population")

    # initializing population
    population = init_population(
        individs=config.population_size,
        in_shape=config.input_format,
        network_min_layers=3,
        network_max_layers=30
    )

    # Training initial population:
    population = workers.start(population, config)
    upload_population(population)

    # Running EA algorithm:
    for generation in range(config.generations):
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
            # TODO: Replace seen_modules:
            # else:
            #     print("    - Sub-module insert for {}".format(selected.ID))
            #     mutated = sub_module_insert(
            #         selected, random_sample(seen_modules), config
            #     )

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
            )
        )

        removable = len(population) - config.population_size
        population, removed = population[removable:], population[:removable]

        # Removing unused models:
        if not config.save_all_results:
            garbage_collector.collect_garbage(removed, population, config)
        upload_population(population)
        print("--> Generation {} Leaderboards:".format(generation))
        generation_finished(population)
        print("--> The following individs were removed by elitism:")
        generation_finished(removed)



def run(config, training_algorithm, job_start_callback, job_end_callback):
    print("\n\nEvolving architecture")
    start_time = time.time()

    evolve_architecture(selection=tournament, config=config)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
