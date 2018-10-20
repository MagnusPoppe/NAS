import gc
import time
import random
from operator import attrgetter

from evolutionary_operations.initialization import init_population
from evolutionary_operations.mutation_for_operators import mutate
from evolutionary_operations.mutation_for_sub_modules import sub_module_insert
from evolutionary_operations.selection import tournament, trash_bad_modules
from output import output_stats, generation_finished_print
from mnist_dataset import mnist_configure


def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

def evolve_architecture(generations, individs, train, fitness, selection, epochs, batch_size, in_shape, classes):
    seen_modules = []

    # initializing population
    population = init_population(individs, in_shape, classes, 3, 8)
    train(population, epochs, batch_size)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))
    seen_modules += population
    for generation in range(generations):

        print("\nGeneration", generation)
        children = []
        print("--> Mutations:")
        for selected in selection(population, individs):
            if random.uniform(0, 1) < 0.2:
                selected = sub_module_insert(selected, random_sample(seen_modules), in_shape, classes, train)
                print("    - Sub-module insert for {}".format(selected.ID))
            else:
                selected = mutate(selected, in_shape, classes)
                print("    - Operation Mutation for {}".format(selected.ID))

            children += [selected]
        # Elitism:
        train(children, epochs, batch_size)
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]

        seen_modules += children
        generation_finished_print(generation, population)

        if generation % 10 == 0:
            seen_modules = trash_bad_modules(seen_modules, evaluate, modules_to_keep=len(population)*2)
            gc.collect()
    return population


if __name__ == '__main__':
    print("Evolving architecture")
    start_time = time.time()
    # nn_input_shape = (784,)
    nn_input_shape = (28, 28, 1)
    train, evaluate = mnist_configure(classes=10, use_2D_input=len(nn_input_shape) > 2)

    popultation = evolve_architecture(
        generations=20,
        individs=2,
        fitness=evaluate,
        train=train,
        selection=tournament,
        epochs=1,
        batch_size=1024,
        in_shape=nn_input_shape,
        classes=10
    )
    print("\n\nTraining complete.")
    output_stats(popultation, start_time)
