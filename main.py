import gc
import time
import random
from operator import attrgetter

from evolutionary_operations.initialization import init_population
from evolutionary_operations.mutation import mutate
from evolutionary_operations.selection import tournament, trash_bad_modules
from output import output_stats, generation_finished_print
from mnist_dataset import mnist_configure


def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

def evolve_architecture(generations, individs, train, fitness, selection, epochs, batch_size, in_shape, classes):
    seen_modules = []

    # initializing population
    population = init_population(individs, in_shape, classes, 1, 5)
    train(population, epochs, batch_size)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))
    seen_modules += population
    for generation in range(generations):

        print("\nGeneration {}".format(generation))
        children = []
        for selected in selection(population, individs):
            selected = mutate(selected, in_shape, classes, modules=seen_modules)
            children += [selected]

        # Elitism:
        train(children, epochs, batch_size)
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]

        seen_modules += children
        generation_finished_print(generation, population)

        if generation % 10 == 0:
            seen_modules = trash_bad_modules(seen_modules, evaluate, modules_to_keep=10)
            gc.collect()
    return population


if __name__ == '__main__':
    print("Evolving architecture")
    start_time = time.time()
    train, evaluate = mnist_configure(classes=10)

    popultation = evolve_architecture(
        generations=50,
        individs=10,
        fitness=evaluate,
        train=train,
        selection=tournament,
        epochs=30,
        batch_size=256,
        in_shape=(784,),
        classes=10
    )
    print("\n\nTraining complete.")
    output_stats(popultation, start_time)
