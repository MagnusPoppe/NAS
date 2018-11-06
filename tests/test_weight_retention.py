import os


os.chdir("..")
import unittest
from copy import deepcopy

from frameworks.keras_decoder import assemble
from modules.dense import Dropout, DenseL, DenseM, DenseS
from modules.module import Module
from evolutionary_operations import mutation_for_operators as mutation_ops
from frameworks.weight_transfer import transfer_predecessor_weights
from datasets.mnist_dataset import mnist_configure

class TestWeightRetention(unittest.TestCase):

    def setUp(self):
        self.module = Module()
        self.module = mutation_ops.append(self.module, DenseS())
        self.module = mutation_ops.append(self.module, DenseM())
        self.module = mutation_ops.append(self.module, DenseL())
        self.module = mutation_ops.append(self.module, Dropout())

    def test_keeps_weights_after_copy(self):
        in_shape = (784,)
        classes = 10


        # Creating keras module:
        self.module.keras_tensor = assemble(self.module, in_shape, classes)

        module_copy = deepcopy(self.module)

        # Precondtion to this copy method, layers in the actual model needs to be the same objects as in the model.
        transfer_predecessor_weights(module_copy, in_shape, classes)
        self.assertEqual(module_copy.keras_operation.layers[1], module_copy.children[0].keras_operation)

        weights, bias = self.module.children[0].keras_operation.get_weights()
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

        # Creating keras module:
        self.module.keras_tensor = assemble(self.module, in_shape, classes)

        module_copy = deepcopy(self.module)
        module_copy = mutation_ops.insert(module_copy, module_copy.children[0], module_copy.children[-1], Dropout())

        transfer_predecessor_weights(module_copy, in_shape, classes)

        weights, bias = self.module.children[0].keras_operation.get_weights()
        copied_weights, copied_bias = module_copy.children[0].keras_operation.get_weights()


        for row, vals in enumerate(weights):
            for col, val in enumerate(vals):
                self.assertEqual(val, copied_weights[row][col], "Weights not the same after copy")

        for i, val in enumerate(bias):
            self.assertEqual(val, copied_bias[i], "Bias not the same after copy")

        train, eval = mnist_configure(classes)
        train([module_copy])
