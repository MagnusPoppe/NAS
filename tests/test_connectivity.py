import unittest

from main import random_sample, operators1D, mutate
from modules.module import Module


class TestModuleConnectivity(unittest.TestCase):

    def test_modules_are_appended_correctly(self):
        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        op1 = random_sample(operators1D)()
        op2 = random_sample(operators1D)()
        op3 = random_sample(operators1D)()
        root += op1
        root += op2
        root += op3

        self.assertTrue(op1 not in op3.prev, "Operations that shouldn't be connected are.")
        self.assertTrue(op2 in op3.prev, "Operation should have been in previous list, but was not.")
        self.assertTrue(op3 in op2.next, "Operation should have been in next list, but was not.")
        self.assertTrue(op1 in op3.prev[0].prev, "Operations in module not fully connected.")

    def test_modules_are_inserted_correctly(self):
            root = Module()
            root.ID = "Root"

            # Adding operations to graph:
            op1 = random_sample(operators1D)()
            op2 = random_sample(operators1D)()
            op3 = random_sample(operators1D)()
            root += op1
            root += op3

            self.assertTrue(op2 not in op1.next and op2 not in op3.prev, "Op2 is in list. Test is broken.")

            # Inserting:
            root.insert(op1, op3, op2)

            self.assertTrue(op2 in op1.next and op2 in op3.prev, "Operation not placed between nodes")
            self.assertTrue(op1 in op3.prev and op3 in op1.next, "Previous structure broke after insert.")

    def test_module_children_fully_connected(self):
        def has_strays(comp, parent, seen):
            if comp in seen:
                return False
            seen += [comp]

            if comp not in parent.children:
                return True
            else:
                return any([has_strays(n, parent, seen) for n in comp.next + comp.prev])


        module = Module()
        for _ in range(40):
            module = mutate(module, training=False, compilation=False)
        module.compile((784, ), 10)

        self.assertFalse(has_strays(module.children[0], module, []), "Graph not fully connected...")

    def test_mutation_operator(self):

        for mutations in [2, 3, 10, 50]:
            individs = [Module() for _ in range(10)]
            for individ in individs:
                for _ in range(mutations):
                    individ = mutate(individ)
                self.assertGreater(
                    a=len(individ.find_last()),
                    b=0,
                    msg="Cycles found after mutating Module {} times...".format(mutations)
                )





if __name__ == '__main__':
    unittest.main()
