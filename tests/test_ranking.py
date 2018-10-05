import unittest

from frameworks.common import rank_children
from helpers import random_sample, operators1D
from modules.dense import DenseS
from modules.module import Module


class TestRanker(unittest.TestCase):

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
