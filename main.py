import time
import random
from operator import attrgetter
from evolutionary_operations.mutation import mutate
from evolutionary_operations.selection import tournament, init_population
from helpers import output_stats
from mnist_dataset import mnist_configure


def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

def evolve_architecture(generations, individs, train, fitness, selection):
    seen_modules = []

    # initializing population
    population = init_population(individs)
    train(population, epochs=30)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))
    seen_modules += population

    for generation in range(generations):
        print("\nGeneration {}".format(generation))
        children = []
        for selected in selection(population, size=individs):
            selected = mutate(selected, modules=seen_modules)
            children += [selected]
            # TODO: crossover

            seen_modules += children

        # Elitism:
        train(children, epochs=30)
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]

        print("--> Population best at generation {}: {}".format(generation, population[-1].fitness))
    return population

if __name__ == '__main__':
    print("Evolving architecture")
    start_time = time.time()
    train, evaluate = mnist_configure(classes=10)
    compile_args = ((784,), 10)

    popultation = evolve_architecture(
        generations=10,
        individs=10,
        fitness=evaluate,
        train=train,
        selection=tournament
    )
    print("\nTraining complete.")
    output_stats(popultation, start_time)
