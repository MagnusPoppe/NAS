import random

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL, Dropout
from modules.module import Module

operators2D =  [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]

def random_sample(operators):
    # Selecting a random operation and creating an instance of it.
    selected = random.randint(0, len(operators) - 1)
    return operators[selected]()


if __name__ == '__main__':
    root = Module()

    # Adding operation to graph:
    root += random_sample(operators1D)
    root += random_sample(operators1D)
    root += random_sample(operators1D)
    root += random_sample(operators1D)

    # Adding an alternative route to the graph
    prev = root.children[2]
    end = root.children[3]
    for i in range(3):
        op = random_sample(operators1D)
        prev.next += [op]
        op.prev += [prev]
        root.children += [op]
        prev = op
    op.next += [end]
    end.prev += [op]

    print(root)
    # root.visualize()
    print(root.compile(input_shape=(784,)).summary())


