import time
import random
from operator import attrgetter
from src.evolutionary_operations.initialization import init_population
from src.evolutionary_operations.mutation_for_operators import mutate
from src.evolutionary_operations.mutation_for_sub_modules import sub_module_insert
from src.evolutionary_operations.selection import tournament
from firebase.upload import create_new_run, update_status, update_fitness
from src.helpers import random_sample
from src.output import generation_finished
from src.jobs import pre_trainer as pretrain

import builtins
builtins.generation = 0

def evolve_architecture(selection, config):
    update_status("Creating initial population")
    seen_modules = []

    # initializing population
    population = init_population(config['population size'], config['input'], config['classes'], 1, 3)

    # Training initial population:
    pretrain.launch_trainers(population, config)
    update_fitness(population)
    seen_modules += population

    # Running EA algorithm:
    for generation in range(config['generations']):
        # Preparation:
        print("\nGeneration", generation)
        builtins.generation = generation
        children = []

        # 'Mutation:
        print("--> Mutations:")
        update_status("Mutating")
        for selected in selection(population, config['population size']):
            if random.uniform(0, 1) < 0.8:
                print("    - Operation Mutation for {}".format(selected.ID))
                children += [ mutate(selected, config['input'], config['classes']) ]
            else:
                print("    - Sub-module insert for {}".format(selected.ID))
                children += [ sub_module_insert(selected, random_sample(seen_modules), config) ]

        # Training new networks:
        children = list(set(children))  # Preventing inbreeding

        # Elitism:
        population += children
        pretrain.launch_trainers(population, config)
        population.sort(key=attrgetter('fitness'))
        population = population[len(population) - config['population size']:]

        seen_modules += children
        generation_finished(generation, population)


def run(config):
    print("\n\nEvolving architecture")
    start_time = time.time()

    # stop_firebase()
    config['run id'] = create_new_run(config)
    evolve_architecture(selection=tournament, config=config)
    print("\n\nTraining complete. Total runtime:", time.time() - start_time)
