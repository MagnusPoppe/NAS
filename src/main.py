import time
import random
from src.evolutionary_operations.initialization import init_population
from src.evolutionary_operations.mutation_for_operators import mutate
from src.evolutionary_operations.mutation_for_sub_modules import sub_module_insert
from src.evolutionary_operations.selection import tournament
from firebase.upload import create_new_run, update_status, upload_population, update_run
from src.helpers import random_sample
from src.output import generation_finished
from src.MOOA import operators as moo
from src.MOOA.NSGA_II import nsga_ii
import src.jobs.TF_launcher as launcher

from src.jobs import scheduler, garbage_collector

import builtins

builtins.generation = 0


def evolve_architecture(selection, config):
    update_status("Creating initial population")

    # initializing population
    population = init_population(
        individs=config["population size"],
        in_shape=config["input"],
        network_min_layers=3,
        network_max_layers=20,
    )

    # Training initial population:
    launcher.run_jobs(population, config)
    upload_population(population)

    # Running EA algorithm:
    for generation in range(config["generations"]):
        # Preparation:
        print("\nGeneration", generation)
        builtins.generation = generation
        children = []

        # Mutation:
        print("--> Mutations:")
        update_status("Mutating")
        for selected in selection(population, config["population size"]):
            draw = random.uniform(0, 1)
            mutated = None
            if draw < 0.5:
                print("    - Operation Mutation for {}".format(selected.ID))
                mutated = mutate(selected)
            else: # elif draw < 0.9:
                print("    - Creating new net randomly")
                mutated = init_population(1, config["input"], 3, 30)[0]
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
        launcher.run_jobs(population, config)
        population = nsga_ii(
            population,
            moo.classification_objectives(config),
            moo.classification_domination_operator(
                moo.classification_objectives(config)
            )
        )

        removable = len(population) - config["population size"]
        population, removed = population[removable:], population[:removable]

        # Removing unused models:
        if not config['keep all results']:
            garbage_collector.collect_garbage(removed, population, config)
        upload_population(population)
        generation_finished(generation, population)


def run(config, training_algorithm, job_start_callback, job_end_callback):
    # scheduler.initialize(
    #     config, training_algorithm, job_start_callback, job_end_callback
    # )
    print("\n\nEvolving architecture")
    start_time = time.time()

    evolve_architecture(selection=tournament, config=config)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
