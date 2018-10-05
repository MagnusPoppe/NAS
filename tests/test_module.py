import unittest
from copy import deepcopy
from operator import attrgetter

from tensorflow import keras

from main import operators1D, mutate, random_sample
from frameworks.module_decoder import assemble, rank_children

from modules.dense import DenseS
from modules.module import Module


class TestModuleCompile(unittest.TestCase):

    def test_module_rank_children(self):
        sub_module = Module(ID="sub-module") \
            .append(operators1D[2]()) \
            .append(operators1D[2]()) \
            .append(operators1D[2]())

        module = Module(ID="Original") \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(sub_module) \
            .append(operators1D[0]())

        module.insert(module.children[1], module.children[2], operators1D[1]()) # -1
        # module.children[2] is an operation with len(prev) == 2
        # module.children[4] is an operation with len(prev) == 1

        rank_children(module)

        counter = 0
        for node in sorted(module.children, key=attrgetter("rank")):
            self.assertEqual(counter, node.rank, "Not sequential ranks in children")
            counter+=1

    def test_module_decode(self):

        # Case of two inputs into one sub-module:
        sub_module1 = Module(ID="sub-module1") \
            .append(operators1D[2]()) \
            .append(operators1D[2]()) \
            .append(operators1D[2]())

        module_1 = Module(ID="Original1") \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(sub_module1) \
            .append(operators1D[0]())

        module_1.insert(module_1.children[1], module_1.children[2], operators1D[1]())
        module_1.keras_tensor = assemble(module_1, in_shape=(784,), classes=10)

        keras.utils.plot_model(module_1.keras_operation, "decode.png")

        sub_module2 = Module(ID="sub-module2") \
            .append(operators1D[2]()) \
            .append(operators1D[2]()) \
            .append(operators1D[2]())

        module_2 = Module("Original2") \
            .append(sub_module2) \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(operators1D[0]())
        module_2.insert(module_2.children[0], module_2.children[2], operators1D[1]())
        module_2.keras_tensor = assemble(module_2, in_shape=(784,), classes=10)

        keras.utils.plot_model(module_2.keras_operation, "decode.png")

        for _ in range(3):
            mutated = mutate(module_2, compilation=False, training=False)

        mutated.keras_tensor = assemble(mutated, in_shape=(784,), classes=10)

        keras.utils.plot_model(mutated.keras_operation, "mutated_decode.png")


    def test_module_copy_value_error_bug(self):
        sub_module1 = Module(ID="sub-module1") \
            .append(operators1D[2]()) \
            .append(operators1D[2]()) \
            .append(operators1D[2]())

        module = Module(ID="Original1") \
            .append(operators1D[0]()) \
            .append(operators1D[0]()) \
            .append(sub_module1) \
            .append(operators1D[0]())

        module.insert(module.children[1], module.children[2], operators1D[1]())
        copied = deepcopy(module)

        copied = copied.insert(
            first_node=copied.children[0],
            second_node=copied.children[3],
            operation=operators1D[2]()
        )
        mutated_copy = deepcopy(copied)

    def test_ranks_with_submodule(self):

        def assert_ranking(module, expected):
            self.assertEqual(len(expected), len(module.children), "Not all children in expected")
            rank_children(module)
            for i, child in enumerate(expected):
                self.assertEqual(i, child.rank, "Was not ranked correctly")


        sub2 = Module(ID="sub-module2").append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())

        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()
        root += Module(ID="sub-module1").append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())

        branch_start = root.children[0]
        branch_op1 = random_sample(operators1D)()
        branch_op2 = sub2
        branch_end = root.children[1]  # This is a module.

        root.insert(branch_start, branch_end, branch_op1)
        root.insert(branch_op1, branch_end, branch_op2)


        expected = [
            root.children[0],
            root.children[3],
            root.children[4],
            root.children[1],
            root.children[2],
        ]
        assert_ranking(root, expected)

        root.remove(branch_op1)
        expected = [
            root.children[0],
            root.children[3],
            root.children[1],
            root.children[2],
        ]
        assert_ranking(root, expected)