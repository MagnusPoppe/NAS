import unittest
import os

os.chdir("..")

from modules.module import Module
from modules import dense


from evolutionary_operations import mutation


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
        cls.module = mutation.append(cls.module, cls.op1)
        cls.module = mutation.append(cls.module, cls.op2)
        cls.module = mutation.append(cls.module, cls.op3)

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
        self.module = mutation.connect(self.module, self.op1, self.op2)
        check_for_duplicates(self, self.op2.prev, "Found duplicate after connect...")
        check_for_duplicates(self, self.op1.next, "Found duplicate after connect...")

        self.assertTrue(
            len(self.op1.next) == len(self.op2.prev) == len(self.op2.next) == len(self.op3.prev) == 1,
            "Number of connections is wrong."
        )
        self.module = mutation.connect(self.module, self.op1, self.op3)
        self.assertIn(self.op3, self.op1.next, "Connection not created...")
        self.assertIn(self.op1, self.op3.prev, "Connection not created...")


    def test_insert(self):
        # Final product should look like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->

        op4 = dense.Dropout()
        self.module = mutation.insert(self.module, self.op3, self.op2, op=op4, between=False)

        self.assertIn(op4, self.module.children, "Op4 not child of module...")
        self.assertIn(op4, self.op2.next, "Op4 was expected to be in next list of op2, but was not.")
        self.assertIn(op4, self.op3.prev, "Op4 was expected to be in prev list of op3, but was not.")
        self.assertIn(self.op2, self.op3.prev, "Op4 was expected to be in prev list of op3, but was not.")
        self.assertIn(self.op3, self.op2.next, "Op4 was expected to be in prev list of op3, but was not.")


    def test_insert_between(self):
        # Final product should look like:
        # op1 -> op2 -> op4 -> op3

        op4 = dense.Dropout()
        self.module = mutation.insert(self.module, self.op2, self.op3, op=op4, between=True)

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

    def test_remove_node_fully_connected(self):
        # Initial model should look like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->
        # Final model:
        # op1 -> op2    ->     op3

        op4 = dense.Dropout()
        self.module = mutation.insert(self.module, self.op2, self.op3, op=op4, between=False)

        self.module = mutation.remove(self.module, op=op4)

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

    def test_remove_last_node(self):
        # Initial model should look like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->
        # Final model:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->
        op4 = dense.Dropout()
        self.module = mutation.insert(self.module, self.op2, self.op3, op=op4, between=False)
        self.module = mutation.remove(self.module, op=self.op3)
        self.assertIn(self.op3, self.module.children, "op3 was removed when illegal to remove.")
        self.assertEqual(len(self.op3.prev), 2, "op3 was disconnected from its previous...")

        # Initial model should look like:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->
        # middel step model:
        # op1 -> op2    ->     op3 -> op5
        #          \          /
        #           -> op4 ->
        # final step model:
        # op1 -> op2    ->     op3
        #          \          /
        #           -> op4 ->
        # Adding a node to the end will make deletion possible:
        op5 = dense.Dropout()
        self.module = mutation.append(self.module, op5)
        self.assertEqual(self.module.find_last()[0], op5, "When appending op5, it was placed incorrectly...")
        self.module = mutation.remove(self.module, op5)
        self.assertEqual(self.module.find_last()[0], self.op3, "Last operation was not op3 as expected after remove.")
        self.assertEqual(len(self.op3.next), 0, "Next-connection not removed from op3.next")
        self.assertEqual(len(self.op3.prev), 2, "previous connecitons for op3 was not maintained")
        self.assertEqual(len(op5.prev), 0, "op5.prev was not emptied...")


    def test_remove_first_node(self):
        # final step model:
        # op2    ->     op3
        #   \          /
        #    -> op4 ->
        # Adding a node to the end will make deletion possible:
        self.module = mutation.remove(self.module, self.op1)
        self.assertEqual(len(self.op1.next), 0, "first node's next was not emptied")
        self.assertEqual(len(self.op2.prev), 0, "New first node has previous ties.")
        self.assertNotIn(self.op1, self.module.children, "Op1 was not removed from children.")
        self.assertEqual(self.module.find_first(), self.op2, "Op2 was expected to be the first node but was not.")

    # def test_stress_test_initialization
    #     pass