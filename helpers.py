import random
import time

from tensorflow import keras

from modules.convolution import Conv3x3, Conv5x5
from modules.dense import DenseS, DenseM, DenseL, Dropout

operators2D = [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]
registered_modules = []

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def random_sample_remove(collection):
    # Selecting a random operation and creating an instance of it.
    # Then deletes the sample
    index = random.randint(0, len(collection) - 1)
    elem = collection.pop(index)
    return elem


def output_stats(population, _time=None, plot_folder="./results"):
    import os
    os.makedirs(plot_folder, exist_ok=True)

    print("--> Accuracy of the best architecture was {} % ({})".format(population[-1].fitness, population[-1].ID))
    print("--> Plots of different network architectures can be found under {}".format(plot_folder))
    if _time:
        print("--> Total elapsed time: {}".format(int(time.time() - _time)))

    def plot_model(individ, img_name):
        keras.utils.plot_model(individ.keras_operation, to_file='{}/{} ({}).png'.format(
            plot_folder, img_name, individ.ID)
       )

    # Find biggest/smallest architecture:
    biggest = None
    smallest = None
    for individ in population:
        if not biggest or len(individ.children) > len(biggest.children):
            biggest = individ
        elif not smallest or len(individ.children) < len(biggest.children):
            smallest = individ

    plot_model(population[0], "lowest_accuracy")
    plot_model(population[-1], "highest_accuracy")
    if smallest: plot_model(smallest, "smallest_architecture")
    if biggest:  plot_model(biggest,  "biggest_architecture")