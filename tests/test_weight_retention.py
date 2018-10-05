import os
os.chdir("..")
import unittest
from copy import deepcopy

from evolutionary_operations.mutation import transfer_predecessor_weights
from frameworks.keras_decoder import assemble
from modules.dense import Dropout, DenseL, DenseM, DenseS
from modules.module import Module

from mnist_dataset import mnist_configure

class TestWeightRetention(unittest.TestCase):

    def test_keeps_weights_after_copy(self):
        in_shape = (784,)
        classes = 10

        module = Module()
        module += DenseS()
        module += DenseM()
        module += DenseL()
        module += Dropout()

        # Creating keras module:
        module.keras_tensor = assemble(module, in_shape, classes)

        module_copy = deepcopy(module)

        # Precondtion to this copy method, layers in the actual model needs to be the same objects as in the model.
        transfer_predecessor_weights(module_copy, in_shape, classes)
        self.assertEqual(module_copy.keras_operation.layers[1], module_copy.children[0].keras_operation)

        weights, bias = module.children[0].keras_operation.get_weights()
        copied_weights, copied_bias = module_copy.children[0].keras_operation.get_weights()


        for row, vals in enumerate(weights):
            for col, val in enumerate(vals):
                self.assertEqual(val, copied_weights[row][col], "Weights not the same after copy")

        for i, val in enumerate(bias):
            self.assertEqual(val, copied_bias[i], "Bias not the same after copy")

        for row, vals in enumerate(weights):
            for col, val in enumerate(vals):
                pass

        train, eval = mnist_configure(classes)
        train([module_copy])

    def test_keeps_weights_after_copy_and_insert(self):
        in_shape = (784,)
        classes = 10

        module = Module()
        module += DenseS()
        module += DenseM()
        module += DenseL()
        module += Dropout()

        # Creating keras module:
        module.keras_tensor = assemble(module, in_shape, classes)

        module_copy = deepcopy(module)
        module_copy.insert(module_copy.children[0], module_copy.children[-1], Dropout())

        transfer_predecessor_weights(module_copy, in_shape, classes)

        weights, bias = module.children[0].keras_operation.get_weights()
        copied_weights, copied_bias = module_copy.children[0].keras_operation.get_weights()


        for row, vals in enumerate(weights):
            for col, val in enumerate(vals):
                self.assertEqual(val, copied_weights[row][col], "Weights not the same after copy")

        for i, val in enumerate(bias):
            self.assertEqual(val, copied_bias[i], "Bias not the same after copy")

        train, eval = mnist_configure(classes)
        train([module_copy])
