from copy import deepcopy
from operator import attrgetter

from tensorflow import keras
from modules.base import Base
from modules.dense import Dropout
from modules.operation import Operation

global_id = 1


class Module(Base):
    """
    Module is a collection of one or more modules and operations
    """

    def __init__(self, ID=""):
        super().__init__()
        self.children = []
        self.ID = ID
        self.keras_operation = None

    def __iadd__(self, other):
        if isinstance(other, Operation) or isinstance(other, Module):
            self.append(other)
        return self

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity on module level. """
        new_mod = Module(self.ID + "_copy")
        new_mod.nodeID = self.nodeID
        new_mod.children += [deepcopy(child) for child in self.children]

        # Copying connectivity for all children:
        for i, child in enumerate(self.children):
            for cn in child.next:
                new_mod.children[i].next += [new_mod.children[self.children.index(cn)]]
            for cp in child.prev:
                new_mod.children[i].prev += [new_mod.children[self.children.index(cp)]]
        return new_mod

    def append(self, op):
        if len(self.children) == 1:
            self.children[0].next += [op]
            op.prev += [self.children[0]]
        elif len(self.children) > 1:
            previous = self.children[-1]
            previous.next += [op]
            op.prev += [previous]

        self.children += [op]
        return self

    def insert(self, first_node, second_node, operation):
        """
        Inserts operation between two nodes.
        :param first_node:
        :param second_node:
        :param operation:
        :return:
        """
        def is_before(node, target):
            if node == target: return True
            elif node.prev: return any([is_before(prev, target) for prev in node.prev])
            else: return False

        # 1. Switch if first_node after second_node (no cycles).
        if is_before(first_node, second_node):
            temp = second_node
            second_node = first_node
            first_node = temp

        # 2. Connect fully.
        first_node.next += [operation]
        operation.prev += [first_node]
        operation.next += [second_node]
        second_node.prev += [operation]
        self.children += [operation]
        return self

    def remove(self, child: Base):
        # Removing from list:
        index = self.children.index(child)
        self.children.pop(index)

        # Cutting ties:
        for p in child.next: p.prev.remove(child)
        for p in child.prev: p.next.remove(child)

    def visualize(self):
        # Local imports. Server does not have TKinter and will crash on load.
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        def draw(prev, current):
            if current.nodeID is None:
                global global_id
                current.nodeID = "{}: {}".format(global_id, current.ID)
                global_id += 1

            if prev:
                G.add_node(current.nodeID)
                G.add_edge(prev.nodeID, current.nodeID)
            else:
                G.add_node(current.nodeID)

            if len(current.prev) <= 1 or all([x.nodeID != None for x in current.prev]):
                for node in current.next:
                    draw(current, node)

        draw(prev=[], current=self.find_first())

        plt.subplot(111)
        nx.draw(G, with_labels=True, arrowsize=1, arrowstyle='fancy')
        plt.show()

    def find_first(self):
        def on(operation):
            if operation.prev: return on(operation.prev[0])
            return operation
        return on(self.children[0])

    def find_last(self):
        def find_end(comp:Base, seen) -> list:
            ends = []
            if comp in seen: return ends
            else:
                seen += [comp]
                if comp.next:
                    for next_module in comp.next:
                        ends += find_end(next_module, seen)
                else:
                    ends += [comp]
                return ends

        return find_end(self.children[0], [])