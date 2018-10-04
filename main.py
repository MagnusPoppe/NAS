import time
import random
from operator import attrgetter
from evolutionary_operations import evaluation
from evolutionary_operations import mutation
from evolutionary_operations.selection import tournament, init_population
from helpers import output_stats
from mnist_dataset import evaluate, train


def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def evolve_architecture(train_func, eval_func, generations, individs, selection, epochs, batch_size, classes):
    seen_modules = []

    # initializing population
    population = init_population(individs)
    evaluation.training(train_func, population, epochs, batch_size, classes)

    # population fitness
    evaluation.evaluation(eval_func, population)
    population.sort(key=attrgetter('fitness'))
    seen_modules += population

    for generation in range(generations):
        print("\nGeneration {}".format(generation))
        children = []
        for selected in selection(population, size=individs):
            selected = mutation.random_mutation(selected, modules=seen_modules)
            children += [selected]
            # TODO: crossover

            seen_modules += children

        # Elitism:
        evaluation.training(train_func, children, epochs, batch_size, classes)
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]

        print("--> Population best at generation {}: {}".format(generation, population[-1].fitness))
    return population


if __name__ == '__main__':
    compile_args = ((784,), 10)

    start_time = time.time()
    print("Evolving architecture")
    popultation = evolve_architecture(
        eval_func=evaluate,
        train_func=train,
        generations=10,
        individs=10,
        selection=tournament,
        epochs=15,
        batch_size=256,
        classes=10
    )
    print("\nTraining complete.")
    output_stats(popultation, start_time)
