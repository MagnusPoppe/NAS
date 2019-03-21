import json
import unittest
import os
from copy import deepcopy

os.chdir("..")

from src.training import cifar10
from src.buildingblocks.ops.convolution import Conv3x3
from src.buildingblocks.ops.pooling import MaxPooling2x2, AvgPooling2x2


from src.frameworks.keras_decoder import assemble
from src.buildingblocks.module import Module
from src.buildingblocks.ops import dense
from src.ea_nas.evolutionary_operations import mutation_for_operators as mutation_ops


class TestAssembly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Neural network setup:
        cls.in_shape = (784,)
        cls.classes = 10

    def setUp(self):
        # Module looks like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->

        self.module = Module()
        self.op1 = dense.DenseS()
        self.op2 = dense.DenseM()
        self.op3 = dense.DenseL()
        self.module = mutation_ops.append(self.module, self.op1)
        self.module = mutation_ops.append(self.module, self.op2)
        self.module = mutation_ops.append(self.module, self.op3)
        self.op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op3, self.op2, op=self.op4, between=False)

        # Setting up sub module for those cases:
        #  first    ->      last
        #    \             /
        #     -> branch ->
        self.sub_module = Module()
        self.first, self.last, self.branch = dense.DenseS(), dense.DenseL(), dense.DenseM()
        self.sub_module = mutation_ops.append(self.sub_module, self.first)
        self.sub_module = mutation_ops.append(self.sub_module, self.last)
        self.sub_module = mutation_ops.insert(self.sub_module, self.first, self.last, self.branch)

    def test_assembles_with_branched_network(self):
        train, _ = mnist_configure(self.classes)
        self.module.keras_operation = assemble(self.module, self.in_shape, self.classes)
        train([self.module], 1, 1024)

    def test_assembles_with_first_layer_as_module(self):
        # Genotype after assembly:
        # sub_module -> op2    ->     op3
        #                 \          /
        #                  -> op4 ->

        self.module = mutation_ops.insert(self.module, self.op1, self.op2, self.sub_module, between=True)
        self.module = mutation_ops.remove(self.module, self.op1)

        train, _ = mnist_configure(self.classes)
        self.module.keras_operation = assemble(self.module, self.in_shape, self.classes)
        train([self.module], 1, 1024)

    def test_assembles_with_pretrained_module_as_first_module(self):
        train, _ = mnist_configure(self.classes)

        self.sub_module.keras_operation = assemble(self.sub_module, self.in_shape, self.classes)
        train([self.sub_module], 1, 1024)

        sub_module_copy = deepcopy(self.sub_module)
        sub_module_copy = mutation_ops.insert(sub_module_copy, sub_module_copy.children[2], sub_module_copy.children[0], dense.Dropout())
        sub_module_copy = mutation_ops.transfer_predecessor_weights(sub_module_copy, self.in_shape, self.classes)

        self.module = mutation_ops.insert(self.module, self.op1, self.op2, self.sub_module, between=True)
        self.module = mutation_ops.insert(self.module, self.sub_module, self.op4, sub_module_copy, between=False)
        self.module = mutation_ops.remove(self.module, self.op1)

        self.module.keras_operation = assemble(self.module, self.in_shape, self.classes)
        train([self.module], 1, 1024)

    def test_assemble_with_pooling_op(self):
        with open("./datasets/cifar10-home-ssh.json", "r") as f:
            config = json.load(f)
        server = config.servers[0]

        individ = Module()
        individ = mutation_ops.append(individ, Conv3x3())
        individ = mutation_ops.append(individ, Conv3x3())
        individ = mutation_ops.append(individ, dense.DenseL())
        individ = mutation_ops.insert(individ, individ.children[1], individ.children[2], dense.DenseL())
        individ = mutation_ops.insert(individ, individ.children[1], individ.children[2], MaxPooling2x2())
        individ = mutation_ops.insert(individ, individ.children[1], individ.children[2], AvgPooling2x2())
        individ = mutation_ops.append(individ, dense.DenseL())

        training, evalutation, name, inputs = cifar10.configure(config.classes_in_classifier, server)

        model = assemble(individ, config.input_format, config.classes_in_classifier)

        training_history = training(
            model=model,
            device="/gpu:0",  # server['device'],
            epochs=0,
            batch_size=1000
        )
