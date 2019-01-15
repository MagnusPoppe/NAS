import unittest
from src.MOOA.NSGA_II import nsga_ii


class CPU:
    def __init__(self, name, cores, threads, speed, power, price):
        self.name = name
        self.cores = cores
        self.threads = threads
        self.speed = speed
        self.power = power
        self.price = price

    def __str__(self):
        return self.name


class TestMutationOperator(unittest.TestCase):

    def setUp(self):
        dataset = [
            {
                "name": "AMD Ryzen Threadripper 1950X",
                "cores": 16,
                "threads": 32,
                "speed": 4.0,
                "power": 180,
                "price": 7999
            },
            {
                "name": "AMD Ryzen 7 1800X",
                "cores":  8,
                "threads": 16,
                "speed": 4.0,
                "power":  95,
                "price": 3189
            },
            {
                "name": "Intel Xeon E5-2620 V4",
                "cores":  8,
                "threads":  8,
                "speed": 2.1,
                "power":  85,
                "price": 4799
            },
            {
                "name": "Intel Core i7-7800X",
                "cores":  6,
                "threads": 12,
                "speed": 4.0,
                "power": 140,
                "price": 3779
            },
            {
                "name": "Intel Core i7-8700",
                "cores":  6,
                "threads": 12,
                "speed": 4.6,
                "power":  65,
                "price": 2999
            },
            {
                "name": "Intel Core i9-7900X",
                "cores": 10,
                "threads": 20,
                "speed": 4.5,
                "power": 140,
                "price": 9726
            }
        ]
        self.test_subjects = [CPU(*data.values()) for data in dataset]

        self.test_objectives = [
            lambda p: (p.price / (p.cores * p.speed)),
            lambda p: (p.power / (p.cores * p.speed)),
            lambda p: p.threads,
        ]

        def domination_operator(p, q) -> bool:
            facts = [
                (q.price / (q.cores * q.speed)) -
                (p.price / (p.cores * p.speed)),
                (q.power / (q.cores * q.speed)) -
                (p.power / (p.cores * p.speed)),
                p.threads - q.threads
            ]
            return any([f >= 0 for f in facts]) and all([f > 0 for f in facts])
        self.domination_operator = domination_operator

    def full_sorting_test(self):
        final = nsga_ii(
            population=self.test_subjects,
            objectives=self.test_objectives,
            domination_operator=self.domination_operator
        )

        self.assertEqual(
            final[0].name == "AMD Ryzen 7 1800X", "Sorted wrong... ")
        self.assertEqual(
            final[1].name == "Intel Core i7-8700", "Sorted wrong... ")
        self.assertEqual(
            final[2].name == "AMD Ryzen Threadripper 1950X", "Sorted wrong... ")
        self.assertEqual(
            final[3].name == "Intel Core i7-7800X", "Sorted wrong... ")
        self.assertEqual(
            final[4].name == "Intel Core i9-7900X", "Sorted wrong... ")
        self.assertEqual(
            final[5].name == "Intel Xeon E5-2620 V4", "Sorted wrong... ")
