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

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]

def init_population(individs=10):
    population = []
    for i in range(individs):
        root = Module()
        for _ in range(random.randint(1, 10)):
            root = mutate(root)
        population += [root]
    return population

def mutate(module:Module) -> Module:
    op = random_sample(operators1D)()
    selected = random.uniform(0,1)
    if selected < 0.5 or len(module.children) <= 3:
        return module.append(op)
    else:
        children = list(range(len(module.children))) # uten tilbakelegging
        return module.insert(
            second_node=module.children[children.pop(random.randint(1, len(children)-2))],
            first_node=module.children[children.pop(random.randint(1, len(children)-2))],
            operation=op
        )

def tournament(population, size):
    selected = []
    individs = np.array(range(len(population)))
    np.random.shuffle(individs)
    individs = individs[:size]
    for i, j in zip(individs[:int(len(individs) / 2)], individs[int(len(individs) / 2):]):
        yield population[i] if (population[i].fitness > population[j].fitness) else population[j]

def mnist_configure(compile_parameters =((784,), True, 10)):
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

    def train(population, epochs=5):
        print("Running training for {} epochs on {} individs".format(epochs, len(population)))
        for individ in population:
            model = individ.compile(*compile_parameters)

            # TODO: FIX THIS...
            if model is None:
                individ.fitness = 0.0
                continue

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
            _, individ.fitness = metrics

    def evaluate(population):
        for individ in population:
            model = individ.compile(*compile_parameters)

            if model is None:
                individ.fitness = 0.0
                continue

            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
            metrics = model.evaluate(x_test, keras.utils.to_categorical(y_test, num_classes=10), verbose=0)
            _, individ.fitness = metrics

    return train, evaluate

def evolve_architecture(generations, individs, train, fitness, selection):
    # TODO: Training as a separate feature...
    # initializing population
    population = init_population(individs)

    # population fitness
    fitness(population)
    population.sort(key=attrgetter('fitness'))

    for generation in range(generations):
        print("Population best at {} generation: {}".format(generation, population[0].fitness))
        train(population, epochs=3)
        children = []
        for selected in selection(population, size=int(individs/2)):

            selected = mutate(selected)  # TODO: Module.copy?
                                         # TODO: crossover
            fitness([selected])          # TODO: Make parallel...
            children += [selected]

        # kill bad children
        population += children
        population.sort(key=attrgetter('fitness'))
        population = population[len(population)-individs:]
    return population[0]

if __name__ == '__main__':
    train, evaluate = mnist_configure()

    best = evolve_architecture(
        generations=100,
        individs=10,
        fitness=evaluate,
        train=train,
        selection=tournament
    )
    keras.utils.plot_model(best.compile((784,), is_root = True, classes = 10), to_file='best_model.png')
