import unittest

from src.buildingblocks.ops.operation import Operation
from src.pattern_nets.initialization import initialize_patterns


def find_islands(pattern):
    """ Looks for completely separate children,
        Where there are no connections inbetween
    """
    def find_members(child, seen):
        if child in seen:
            return []
        members = [child]
        seen += [child]
        for p in child.prev:
            members += find_members(p, seen)
        for n in child.next:
            members += find_members(n, seen)
        return members

    islands = []
    seen = []
    for child in pattern.children:
        if child in seen:
            continue
        members = find_members(child, [])
        islands += [members]
        seen += members
    return islands


def find_all_behind(op, seen):
    if op in seen:
        return []
    seen += [op]
    behind = []
    for prev in op.prev:
        behind += [prev] + find_all_behind(prev, seen)
    return behind



class TestPatterns(unittest.TestCase):

    def setUp(self):
        from src.buildingblocks.module import reset_naming
        reset_naming()

    # @unittest.skip("Working on test-3")
    def test_1_initialization(self):
        from src.ea_nas.evolutionary_operations.mutation_for_operators import is1D, is2D
        def discover_cycle(current, i, max_rounds):
            high = 0
            if i >= max_rounds:
                return i
            if current.next:
                for nex in current.next:
                    high = max(i, discover_cycle(nex, i+1, max_rounds))
            return high

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
                self.assertLess(discover_cycle(op, 0, 6), 6, "Has cycles... ")


    def test_2_recombination(self):
        from src.ea_nas.evolutionary_operations.mutation_for_operators import is2D
        from src.pattern_nets import combiner
        patterns = initialize_patterns(count=1000)
        nets = combiner.combine(patterns, num_nets=50, min_size=10, max_size=100)

        for net in nets:
            self.assertGreater(len(net.children), 0, "Should be operations in the nets.")
            if len(net.children) > 1:
                islands = find_islands(net)
                self.assertTrue(
                    all(x in net.children for island in islands for x in island),
                    "This network is connected to operations which are not children of this network... "
                )
            for op in net.children:
                self.assertTrue(isinstance(op, Operation), "Only operations in net, no modules or patterns...")
                if is2D(op):
                    behind = find_all_behind(op, [])
                    self.assertTrue(all([is2D(b) for b in behind]), "No 1D before 2D")

    def test_3_converts_to_keras_model(self):
        from src.pattern_nets import combiner
        patterns = initialize_patterns(count=1000)
        nets = combiner.combine(patterns, num_nets=1, min_size=20, max_size=59)

        for net in nets:
            from src.frameworks.keras import module_to_model
            from tensorflow import keras
            model = module_to_model(net, [32, 32, 3], 10)
            keras.utils.plot_model(model, to_file=f"tests/output/{net.ID}.png")
            shape = model.output.shape

            # Checking for model outputs if they are shaped correctly:
            self.assertEqual(len(shape), 2, "Wrong shape of returned shape...")
            self.assertTrue(shape[0].value is None, "Shape part 0 should be None...")
            self.assertEqual(shape[1].value, 10, "Shape part 2 should match classes...")
