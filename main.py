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

def evolve_architecture(generations, individs, train, fitness, selection, epochs, batch_size):
    seen_modules = []

    # initializing population
    population = init_population(individs)
    train(population, epochs, batch_size)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))
    seen_modules += population
    try:
        for generation in range(generations):
            print("\nGeneration {}".format(generation))
            children = []
            for selected in selection(population, size=individs):
                selected = mutate(selected, modules=seen_modules)
                children += [selected]
                # TODO: crossover

                seen_modules += children

            # Elitism:
            train(children, epochs, batch_size)
            population += children
            population.sort(key=attrgetter('fitness'))
            population = population[len(population)-individs:]

            print("--> Generation {} Results: \n"
                  "    - Best: {} % Accuracy ({})\n"
                  "    - Runner up: {} % Accuracy ({})"
                  .format(generation,
                          population[-1].fitness, population[-1].ID,
                          population[-2].fitness, population[-2].ID)
              )
    except KeyboardInterrupt as e: pass
    finally: return population

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
        selection=tournament,
        epochs=30,
        batch_size=256
    )
    print("\n\nTraining complete.")
    output_stats(popultation, start_time)
