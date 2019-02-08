import time
import random
from operator import attrgetter
from src.evolutionary_operations.initialization import init_population
from src.evolutionary_operations.mutation_for_operators import mutate
from src.evolutionary_operations.mutation_for_sub_modules import sub_module_insert
from src.evolutionary_operations.selection import tournament
from firebase.upload import create_new_run, update_status, upload_population, update_run
from src.helpers import random_sample
from src.output import generation_finished
from src.MOOA import operators as moo
from src.MOOA.NSGA_II import nsga_ii

# from src.jobs import pre_trainer as pretrain
from src.jobs import scheduler

import builtins

builtins.generation = 0


def evolve_architecture(selection, config):
    update_status("Creating initial population")
    seen_modules = []

    # initializing population
    population = init_population(
        individs=config["population size"],
        in_shape=config["input"],
        network_min_layers=3,
        network_max_layers=10,
    )

    # Training initial population:
    scheduler.queue_all(population, config)
    scheduler.await_all_jobs_finish()
    upload_population(population)
    seen_modules += population

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
            if draw < 0.6:
                print("    - Operation Mutation for {}".format(selected.ID))
                mutated = mutate(selected)
            elif draw < 0.9:
                print("    - Creating new net randomly")
                mutated = init_population(1, config["input"], 3, 30)[0]
            else:
                print("    - Sub-module insert for {}".format(selected.ID))
                mutated = sub_module_insert(
                    selected, random_sample(seen_modules), config
                )

            if mutated:
                scheduler.queue(mutated, config)
                children += [mutated]

        # Training networks:
        children = list(children)  # Preventing inbreeding
        scheduler.queue_all(population, config)
        scheduler.await_all_jobs_finish()  # Blocking...

        # Elitism:
        population += children
        population = nsga_ii(population, moo.objectives(), moo.domination_operator)
        population = population[len(population) - config["population size"] :]

        upload_population(population)
        seen_modules += children
        generation_finished(generation, population)


def run(config, training_algorithm, job_start_callback, job_end_callback):
    scheduler.initialize(
        config, training_algorithm, job_start_callback, job_end_callback
    )
    print("\n\nEvolving architecture")
    start_time = time.time()

    evolve_architecture(selection=tournament, config=config)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
