import random

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL, Dropout
from modules.module import Module

if __name__ == '__main__':

    all_operators = [DenseS, DenseM, DenseL, Dropout] # Conv3x3, Conv5x5,

    root = Module()

    for _ in range(3):
        # selecting what list to pick operation from:
        operations = all_operators if len(root.children) < 1 else root.children[-1].compatible

        # Selecting a random operation and creating an instance of it.
        selected = random.randint(0, len(all_operators)-1)
        op = all_operators[selected]()

        # Adding operation to graph:
        root += op

    print(root)
    print(root.compile(input_shape=(784,)))

