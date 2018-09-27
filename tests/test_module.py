import unittest
from copy import copy

from main import random_sample, operators1D
from modules.module import Module


class TestModuleCompile(unittest.TestCase):

    def test_module_copy(self):
        original = Module(ID="Original") \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(operators1D[0]())

        original.insert(original.children[1], original.children[2], operators1D[1]())
        original.insert(original.children[3], original.children[2], operators1D[1]())
        original.insert(original.children[4], original.children[2], operators1D[1]())

        cp = copy(original)

        original.compile((784,), 10)
        cp.compile((784,), 10)

        from tensorflow import keras

        keras.utils.plot_model(original.keras_operation, to_file=original.ID+".png")
        keras.utils.plot_model(cp.keras_operation, to_file=cp.ID+".png")

