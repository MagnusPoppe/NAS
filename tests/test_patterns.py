import os
os.chdir("..")
import unittest
from src.pattern.initialization import initialize_patterns

class TestPatterns(unittest.TestCase):

    def test_initialization(self):
        patterns = initialize_patterns(count=1550)
        for pattern in patterns:
            self.assertIn(pattern.type, ["1D", "2D"], "Not correct type of pattern")
            self.assertTrue(1 < pattern.layers <= 4, "Too few or many layers in pattern")
            self.assertTrue(1 < len(pattern.children) <= 4, "Wrong number of children.")

        # names = [pattern.name for pattern in patterns]
        # while 1 < len(names):
        #     name = names.pop(0)
        #     self.assertTrue(name not in names, "Found duplicate name...")

    def test_recombination(self):
        from src.pattern import recombination
        patterns = initialize_patterns(count=10)

