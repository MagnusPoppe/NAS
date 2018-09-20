import random

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL
from modules.dropout import Dropout
from modules.module import Module

if __name__ == '__main__':

    all_operators = [Conv3x3, Conv5x5, DenseS, DenseM, DenseL, Dropout]
    root = Module()

    for _ in range(3):

        # Selecting a random operation and creating an instance of it.
        selected = random.randint(0, len(all_operators)-1)
        op = all_operators[selected]()

        # Adding operation to graph:
        root += op

    print(root)