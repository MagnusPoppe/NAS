import unittest
import os
os.chdir("..")

from evolutionary_operations.initialization import init_population


class TestModuleCompile(unittest.TestCase):
    for _ in range(10):
        population = init_population(100, (784,), 10, 1, 100)
