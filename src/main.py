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
# from src.jobs import pre_trainer as pretrain
from src.jobs import scheduler

import builtins
builtins.generation = 0

def evolve_architecture(selection, config):
    update_status("Creating initial population")
    seen_modules = []

    # initializing population
    population = init_population(
        individs=config['population size'],
        in_shape=config['input'],
        network_min_layers=3,
        network_max_layers=10
    )

    # Training initial population:
    scheduler.queue_all(population, config)
    scheduler.await_all_jobs_finish()
    upload_population(population)
    seen_modules += population

    # Running EA algorithm:
    for generation in range(config['generations']):
        # Preparation:
        print("\nGeneration", generation)
        builtins.generation = generation
        children = []

        # Mutation:
        print("--> Mutations:")
        update_status("Mutating")
        for selected in selection(population, config['population size']):
            if random.uniform(0, 1) < 0.8:
                print("    - Operation Mutation for {}".format(selected.ID))
                mutated = mutate(selected)
            else:
                print("    - Sub-module insert for {}".format(selected.ID))
                mutated = sub_module_insert(selected, random_sample(seen_modules), config)

            if mutated:
                scheduler.queue(mutated, config)
                children += [mutated]

        # Training networks:
        children = list(children)  # Preventing inbreeding
        scheduler.queue_all(population, config)
        scheduler.await_all_jobs_finish()

        # Elitism:
        population += children
        population.sort(key=lambda ind: ind.fitness[-1])
        population = population[len(population) - config['population size']:]

        upload_population(population)
        seen_modules += children
        generation_finished(generation, population)


def run(config, training_algorithm, job_start_callback, job_end_callback):
    scheduler.initialize(config, training_algorithm, job_start_callback, job_end_callback)
    print("\n\nEvolving architecture")
    start_time = time.time()

    status = "Running"
    try:
        evolve_architecture(selection=tournament, config=config)
        print("\n\nTraining complete. Total runtime:", time.time() - start_time)
        status = "Finished"
    except Exception as e:
        status = "Crashed!"
        raise e
    finally:
        update_run(config, status)
