import os

os.chdir("..")
import unittest
from src.pattern_nets.initialization import initialize_patterns


class TestPatterns(unittest.TestCase):

    def setUp(self):
        from src.buildingblocks.module import reset_naming
        reset_naming()

    def test_1_initialization(self):
        from src.ea_nas.evolutionary_operations.mutation_for_operators import is1D, is2D
        def discover_cycle(current):
            if current.next:
                for nex in current.next:
                    discover_cycle(nex)

        patterns = initialize_patterns(count=20)
        for pattern in patterns:
            self.assertIn(pattern.type, ["1D", "2D"], "Not correct type of pattern")
            self.assertTrue(1 < pattern.layers <= 4, "Too few or many layers in pattern")
            self.assertTrue(1 < len(pattern.children) <= 4, "Wrong number of children.")
            for op in pattern.children:
                if pattern.type == "1D":
                    self.assertTrue(is1D(op))
                if pattern.type == "2D":
                    self.assertTrue(is2D(op))

            # Has no cycles:


    def test_2_recombination(self):
        from src.ea_nas.evolutionary_operations.mutation_for_operators import is1D, is2D
        from src.pattern_nets import recombination
        patterns = initialize_patterns(count=1000)
        nets = recombination.combine_random(patterns, num_nets=50)

        for net in nets:
            self.assertGreater(len(net.children), 0, "Should be operations in the nets.")
            if len(net.children) > 1:
                for module in net.children:
                    self.assertTrue(module.next or module.prev, "No dangling operations in nets...")

    def test_3_converts_to_keras_model(self):
        pass
