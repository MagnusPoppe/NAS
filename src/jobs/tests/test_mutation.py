import unittest
import os

from src.buildingblocks.ops.convolution import Conv3x3, Conv5x5

os.chdir("..")

# from evolutionary_operations.initialization import init_population
from src.buildingblocks.module import Module
from src.buildingblocks.ops import dense

from src.ea_nas.evolutionary_operations import mutation_for_operators as mutation, mutation_operators as mutation_ops


def check_for_duplicates(self, connection_list, msg):
    counter = {}
    for node in connection_list:
        self.assertNotIn(node.ID, counter, msg)
        counter[node] = 1

class TestMutationOperator(unittest.TestCase):

    def setUp(cls):
        cls.module = Module()
        cls.op1 = dense.DenseS()
        cls.op2 = dense.DenseM()
        cls.op3 = dense.DenseL()
        cls.module = mutation_ops.append(cls.module, cls.op1)
        cls.module = mutation_ops.append(cls.module, cls.op2)
        cls.module = mutation_ops.append(cls.module, cls.op3)

    def test_possible_insertion_points(self):
        conv = Conv3x3()
        after, before = mutation.get_possible_insertion_points(self.module, operation=conv)
        self.assertEqual(len(after), 0, "Conv 2D could be inserted after a 1D layer or input.")
        self.assertEqual(len(before), len(self.module.children)-1, "Should be able to be inserted before any except first.")

        # Network setup:
        # 2D input -> conv1     ->      op1 -> op2
        #                 \            /
        #                  -> conv2 ->
        module = Module()
        conv1 = Conv5x5()
        conv2 = Conv3x3()
        op1 = dense.DenseS()
        op2 = dense.DenseL()
        module = mutation_ops.append(module, conv1)
        module = mutation_ops.append(module, op1)
        module = mutation_ops.append(module, op2)
        module = mutation_ops.insert(module, first=conv1, last=op1, op=conv2)

        after, before = mutation.get_possible_insertion_points(module, operation=conv)
        self.assertIn(conv1, after, "Should be able to insert after conv1")
        self.assertIn(conv2, after, "Should be able to insert after conv2")
        self.assertEqual(len(after), 2, "More nodes in after than it should be...")
        self.assertNotIn(conv1, before, "Should not be able to insert before first node.")

        op3 = dense.DenseM()
        after, before = mutation.get_possible_insertion_points(module, operation=op3)
        self.assertEqual(len(after), 3, "Should have been able to insert after every node except last.")
        self.assertIn(op1, after, "Op1 should have been in after list.")
        self.assertIn(op1, before, "Op1 should have been in before list.")
        self.assertIn(op2, before, "Op2 should have been in before list.")


    def test_voting_system(self):
        weights = [("#1", 10), ("#2", 10), ("#3", 5)]
        expected_number_of_votes = sum(x for _, x in weights)

        votes = mutation._generate_votes(weights)
        self.assertEqual(len(votes), expected_number_of_votes, "Wrong number of votes")
        counter = {}
        for vote in votes:
            if vote not in counter:
                counter[vote] = 0
            counter[vote] += 1

        expected_probability = weights[0][1] / expected_number_of_votes
        self.assertEqual(counter["#1"]/len(votes), expected_probability, "Distrubution of votes incorrect")
        self.assertEqual(counter["#1"]/len(votes), counter["#2"]/len(votes), "Probability for #1 not same as #2")
        self.assertGreater(counter["#1"]/len(votes), counter["#3"]/len(votes), "Probability for #1 less than #3")

    def test_connect(self):
        self.module = mutation_ops.connect(self.module, self.op1, self.op2)
        check_for_duplicates(self, self.op2.prev, "Found duplicate after connect...")
        check_for_duplicates(self, self.op1.next, "Found duplicate after connect...")

        self.assertTrue(
            len(self.op1.next) == len(self.op2.prev) == len(self.op2.next) == len(self.op3.prev) == 1,
            "Number of connections is wrong."
        )
        self.module = mutation_ops.connect(self.module, self.op1, self.op3)
        self.assertIn(self.op3, self.op1.next, "Connection not created...")
        self.assertIn(self.op1, self.op3.prev, "Connection not created...")


    def test_insert(self):
        # Final product should look like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->

        op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op3, self.op2, op=op4, between=False)

        self.assertIn(op4, self.module.children, "Op4 not child of module...")
        self.assertIn(op4, self.op2.next, "Op4 was expected to be in next list of op2, but was not.")
        self.assertIn(op4, self.op3.prev, "Op4 was expected to be in prev list of op3, but was not.")
        self.assertIn(self.op2, self.op3.prev, "Op4 was expected to be in prev list of op3, but was not.")
        self.assertIn(self.op3, self.op2.next, "Op4 was expected to be in prev list of op3, but was not.")


    def test_insert_between(self):
        # Final product should look like:
        # op1 -> op2 -> op4 -> op3

        op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op2, self.op3, op=op4, between=True)

        self.assertIn(op4, self.module.children, "Op4 not child of module...")
        self.assertIn(op4, self.op2.next, "Op4 was expected to be in next list of op2, but was not.")
        self.assertIn(op4, self.op3.prev, "Op4 was expected to be in prev list of op3, but was not.")
        self.assertNotIn(self.op2, self.op3.prev, "Op4 was not expected to be in prev list of op3, but was.")
        self.assertNotIn(self.op3, self.op2.next, "Op4 was not expected to be in prev list of op3, but was.")
        self.assertTrue(len(op4.prev) == len(op4.next) == 1, "Wrong number of elements in op4 connections.")

        for i, op in enumerate(self.module.children[:-1]):
            expected_next = 1 if i < len(self.module.children[:-1]) -1 else 0
            self.assertEqual(len(op.next), expected_next, "Next list contained wrong number of children")

            expected_prev = 1 if i > 0 else 0
            self.assertEqual(len(op.prev), expected_prev, "Next list contained wrong number of children")

    def test_remove_node_single_connections(self):
        # Initial model:                   Final model:
        # op1 -> op2    ->     op3         op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->

        op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op2, self.op3, op=op4, between=False)

        self.module = mutation_ops.remove(self.module, op=op4)

        # Testing each of the connections manually:
        self.assertEqual(len(self.op1.prev), 0, "self.op1.prev should have no connections...")
        self.assertEqual(len(self.op1.next), 1, "self.op1.next is connected to more nodes than it should...")
        self.assertEqual(len(self.op2.prev), 1, "self.op2.prev is connected to more nodes than it should...")
        self.assertEqual(len(self.op2.next), 1, "self.op2.next is connected to more nodes than it should...")
        self.assertEqual(len(self.op3.prev), 1, "self.op3.prev is connected to more nodes than it should...")
        self.assertEqual(len(self.op3.next), 0, "self.op3.next should have no connections...")
        self.assertEqual(len(op4.prev), 0, "self.op4.prev should have no connections...")
        self.assertEqual(len(op4.next), 0, "self.op4.next should have no connections...")

        # Testing which nodes is connected to which:
        self.assertNotIn(op4, self.module.children, "Op4 is child of module after remove...")
        self.assertNotIn(op4, self.op3.prev, "Op4 is previous node of op3 after remove...")
        self.assertNotIn(op4, self.op2.next, "Op4 is next node for op2 after remove...")
        self.assertNotIn(self.op2, op4.prev, "Op4 still connected to its next node...")
        self.assertNotIn(self.op3, op4.next, "Op4 still connected to its previous node...")

        # Testing for duplicate connections:
        check_for_duplicates(self, self.op3.prev, "Found duplicate in list self.op3.prev...")
        check_for_duplicates(self, self.op2.next, "Found duplicate in list self.op2.next...")

    def test_remove_node_multi_connections(self):
        # Initial model:                   Final model:
        # op1 -> op2    ->     op3         op1     ->     op3
        #          \          /               \          /
        #           -> op4 ->                  -> op4 ->

        # Setup:
        op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op2, self.op3, op=op4, between=False)

        # Performing action:
        self.module = mutation_ops.remove(self.module, op=self.op2)

        # Testing if all connected nodes children of module
        for child in self.module.children:
            for node in child.prev:
                self.assertIn(node, self.module.children)
            for node in child.next:
                self.assertIn(node, self.module.children)

        # Testing each of the connections manually:
        self.assertEqual(len(self.op1.prev), 0, "Wrong number of nodes connected to op1.prev")
        self.assertEqual(len(self.op1.next), 2, "Wrong number of nodes connected to op1.next")
        self.assertEqual(len(op4.prev), 1, "Wrong number of nodes connected to op4.prev")
        self.assertEqual(len(op4.next), 1, "Wrong number of nodes connected to op4.next")
        self.assertEqual(len(self.op3.prev), 2, "Wrong number of nodes connected to op3.prev")
        self.assertEqual(len(self.op3.next), 0, "Wrong number of nodes connected to op3.next")

        self.assertEqual(len(self.op2.next) + len(self.op2.prev), 0, "Removed node was not emptied")

    def test_remove_last_node(self):
        # Initial model should look like:     Final model (no changes):
        # op1 -> op2    ->     op3            op1 -> op2    ->     op3
        #          \          /                        \          /
        #           -> op4 ->                           -> op4 ->

        op4 = dense.Dropout()
        self.module = mutation_ops.insert(self.module, self.op2, self.op3, op=op4, between=False)
        self.module = mutation_ops.remove(self.module, op=self.op3)
        self.assertIn(self.op3, self.module.children, "op3 was removed when illegal to remove.")
        self.assertEqual(len(self.op3.prev), 2, "op3 was disconnected from its previous...")

        # Initial model:                 middel step model:                 final step model:
        # op1 -> op2    ->     op3       op1 -> op2    ->     op3 -> op5    op1 -> op2    ->     op3
        #          \          /                   \          /                       \          /
        #           -> op4 ->                      -> op4 ->                          -> op4 ->

        # Adding a node to the end will make deletion possible:
        op5 = dense.Dropout()
        self.module = mutation_ops.append(self.module, op5)
        self.assertEqual(self.module.find_last()[0], op5, "When appending op5, it was placed incorrectly...")
        self.module = mutation_ops.remove(self.module, op5)
        self.assertEqual(self.module.find_last()[0], self.op3, "Last operation was not op3 as expected after remove.")
        self.assertEqual(len(self.op3.next), 0, "Next-connection not removed from op3.next")
        self.assertEqual(len(self.op3.prev), 2, "previous connecitons for op3 was not maintained")
        self.assertEqual(len(op5.prev), 0, "op5.prev was not emptied...")


    def test_remove_first_node(self):
        # Initial model:                     final step model:
        # op1 -> op2    ->     op3           op2    ->     op3
        #          \          /                \          /
        #           -> op4 ->                   -> op4 ->

        self.module = mutation_ops.remove(self.module, self.op1)
        self.assertEqual(len(self.op1.next), 0, "first node's next was not emptied")
        self.assertEqual(len(self.op2.prev), 0, "New first node has previous ties.")
        self.assertNotIn(self.op1, self.module.children, "Op1 was not removed from children.")
        self.assertEqual(self.module.find_first(), self.op2, "Op2 was expected to be the first node but was not.")

    # def test_stress_test_initialization(self):
    #     for _ in range(0, 10, 1):
    #         population = init_population(100, (784,), 10, 1, 100)
#
    #         for individ in population:
    #             # Testing if all connected nodes children of module
    #             for child in individ.children:
    #                 for node in child.prev:
    #                     self.assertIn(node, individ.children, "Not fully connected!")
    #                 for node in child.next:
    #                     self.assertIn(node, individ.children, "Not fully connected!")
