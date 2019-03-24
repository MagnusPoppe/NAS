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
        from src.pattern_nets.connecter import find_islands
        patterns = initialize_patterns(count=1000)
        nets = recombination.combine_random(patterns, num_nets=50, max_size=50)

        for net in nets:
            self.assertGreater(len(net.children), 0, "Should be operations in the nets.")
            if len(net.children) > 1:
                islands = find_islands(net)
                self.assertTrue(
                    all(x in net.children for island in islands for x in island),
                    "This network is connected to operations which are not children of this network... "
                )

    def test_3_converts_to_keras_model(self):
        import pickle, multiprocessing as mp
        from src.pattern_nets import recombination
        patterns = initialize_patterns(count=5)
        nets = recombination.combine_random(patterns, num_nets=2, max_size=3)

        for net in nets:
            from src.frameworks.keras import module_to_model
            model = module_to_model(net, [32, 32, 3], 10)
            shape = model.output.shape
            # Checking for model outputs if they are shaped correctly:
            self.assertEqual(len(shape), 2, "Wrong shape of returned shape...")
            self.assertTrue(shape[0] is None, "Shape part 0 should be None...")
            self.assertEqual(shape[1], 10, "Shape part 2 should match classes...")

def parallel_reciever(args):
    import pickle
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
    from src.frameworks.keras import module_to_model
    module, in_shape, classes = pickle.loads(args)
    model = module_to_model(module, in_shape, classes)
    print("=", end="", flush=True)
    return [shape.value for shape in model.output.shape]
