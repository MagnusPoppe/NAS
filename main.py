import copy
import random
from operator import attrgetter

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL, Dropout
from modules.module import Module
from tensorflow import keras
import numpy as np

operators2D = [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]
registered_modules = []

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

def init_population(individs=10, compile_args=((784,), 10)):
    population = []
    for i in range(individs):
        root = Module()
        for _ in range(random.randint(1, 10)):
            root = mutate(root, compilation=False)
        root.compile(*compile_args)
        population += [root]
    return population

def mutate(module:Module, compilation=True, compile_parameters =((784,), 10), training=True) -> Module:
    global registered_modules

    # Selecting what module to mutate in:
    if random.uniform(0,1) < 0.9 or not registered_modules:
        op = random_sample(operators1D)()
    else:
        op = random_sample(registered_modules)

    # Selecting where to place operator:
    selected = random.uniform(0,1)
    if selected < 0.3 or len(module.children) <= 3:
        mutated = module.append(op)
    elif selected < 0.6:
        children = list(range(0, len(module.children))) # uten tilbakelegging
        mutated = module.insert(
            first_node=module.children[children.pop(random.randint(0, len(children)-1))],
            second_node=module.children[children.pop(random.randint(0, len(children)-1))],
            operation=op
        )
    elif selected < 0.8:
        mutated = module # TODO: Copy...
        mutated.remove(random_sample(module.children))
    else:
        mutated = module # TODO: Copy...
        mutated.compile(*compile_parameters)
        if training:
            module.fitness = train([module])

    # Compiles keras model from module:
    if compilation:
        mutated.compile(*compile_parameters)
    return mutated

def tournament(population, size):
    individs = np.array(range(len(population)))
    np.random.shuffle(individs)
    individs = individs[:size]
    for i, j in zip(individs[:int(len(individs) / 2)], individs[int(len(individs) / 2):]):
        yield population[i] if (population[i].fitness > population[j].fitness) else population[j]

def mnist_configure(): # -> (function, function):
    def fix(data):
        return np.reshape(data, (len(data), 784))

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # VALIDATION DATA:
    x_val = fix(x_train[50000:])
    y_val = y_train[50000:]
    x_val = x_val.astype('float32')
    x_val /= 255

    # TRAINING DATA:
    x_train = fix(x_train[:50000])
    y_train = y_train[:50000]
    x_train = x_train.astype('float32')
    x_train /= 255

    # TEST DATA:
    x_test = fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy

    def train(population: list, epochs=5):
        print("Running training for {} epochs on {} models".format(epochs, len(population)))
        for individ in population:
            model = individ.keras_operation
            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

            # RUNNING TRAINING:
            metrics = model.fit(
                x_train,
                keras.utils.to_categorical(y_train, num_classes=10),
                epochs=epochs,
                batch_size=64,
                verbose=0,
                validation_data=(
                    x_val, keras.utils.to_categorical(y_val, num_classes=10)
                )
            )
            individ.fitness = metrics.history['acc'][-1]

    def evaluate(population: list):
        print("Evaluating {} models".format(len(population)))
        for individ in population:
            model = individ.keras_operation
            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
            metrics = model.evaluate(x_test, keras.utils.to_categorical(y_test, num_classes=10), verbose=0)
            individ.fitness = metrics[1]  # Accuracy

    return train, evaluate

def evolve_architecture(generations, individs, train, fitness, selection):
    global registered_modules

    # initializing population
    population = init_population(individs)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))

    for generation in range(generations):
        print("Population best at {} generation: {}".format(generation, population[0].fitness))
        train(population, epochs=3)
        registered_modules += [individ for individ in population if individ not in registered_modules]
        children = []
        for selected in selection(population, size=int(individs/2)):
            selected = mutate(selected)  # TODO: Module.copy before mutation?
                                         # TODO: crossover
            fitness([selected])          # TODO: Make parallel...
            children += [selected]

        # kill bad children
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]
    return population[0]

if __name__ == '__main__':
    print("Evolving architecture")
    train, evaluate = mnist_configure()

    best = evolve_architecture(
        generations=100,
        individs=10,
        fitness=evaluate,
        train=train,
        selection=tournament
    )
    keras.utils.plot_model(best.compile((784,), classes=10), to_file='best_model.png')
