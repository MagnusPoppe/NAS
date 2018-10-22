import gc
import time
import random
from operator import attrgetter

from evolutionary_operations.initialization import init_population
from evolutionary_operations.mutation_for_operators import mutate
from evolutionary_operations.mutation_for_sub_modules import sub_module_insert
from evolutionary_operations.selection import tournament, trash_bad_modules
from firebase.upload import create_new_run, update_status, upload_modules, update_fitness, dont_upload_anything
from output import output_stats, generation_finished
from mnist_dataset import mnist_configure

import builtins
builtins.generation = 0

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def evolve_architecture(generations, individs, train, fitness, selection, epochs, batch_size, in_shape, classes):
    update_status("Creating initial population")
    seen_modules = []

    # initializing population
    population = init_population(individs, in_shape, classes, 3, 8)
    upload_modules(population)
    update_status("Training initial population")
    train(population, epochs, batch_size)
    update_fitness(population)
    seen_modules += population
    for generation in range(generations):
        print("\nGeneration", generation)
        builtins.generation = generation
        children = []
        print("--> Mutations:")
        update_status("Mutating")
        for selected in selection(population, individs):
            if random.uniform(0, 1) < 0.2:
                update_status("Inserting sub-module into module {}".format(selected.ID))
                print("    - Sub-module insert for {}".format(selected.ID))
                selected = sub_module_insert(selected, random_sample(seen_modules), in_shape, classes, train)
                update_status("Mutating")
            else:
                selected = mutate(selected, in_shape, classes)
                print("    - Operation Mutation for {}".format(selected.ID))

            children += [selected]
        # Elitism:
        update_status("Training new children")
        children = list(set(children))  # Preventing inbreeding
        train(children, epochs, batch_size)

        update_status("Elitism")
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population) - individs:]

        seen_modules += children
        update_status("Finishing up")
        generation_finished(generation, population)

        if generation % 10 == 0:
            seen_modules = trash_bad_modules(seen_modules, evaluate, modules_to_keep=len(population) * 2)
            gc.collect()
    return population


def main():
    print("Evolving architecture")
    start_time = time.time()
    # nn_input_shape = (784,)
    nn_input_shape = (28, 28, 1)
    classes = 10
    epochs = 30
    batch_size = 256
    generations = 20
    population_size = 8

    train, evaluate, dataset_name = mnist_configure(
        classes=classes,
        use_2D_input=len(nn_input_shape) > 2
    )

    create_new_run(dataset_name, epochs, batch_size, generations, population_size)
    popultation = evolve_architecture(
        generations=generations,
        individs=population_size,
        fitness=evaluate,
        train=train,
        selection=tournament,
        epochs=epochs,
        batch_size=batch_size,
        in_shape=nn_input_shape,
        classes=classes
    )
    print("\n\nTraining complete.")
    output_stats(popultation, start_time)


if __name__ == '__main__':
    import os
    if "EA-NAS-PROD" in os.environ and os.environ["EA-NAS-PROD"] == 1:
        try:
            main()
        except Exception as e:
            builtins.generation = -1
            update_status(u"Failed: " + str(e))
            raise e
    else:
        dont_upload_anything()
        main()