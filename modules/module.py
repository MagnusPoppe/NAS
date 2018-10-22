import json

from helpers import random_sample_remove, random_sample
from modules.base import Base
from modules.operation import Operation

global_id = 1

def load_name_list():
    with open("./resources/names.json", "r", encoding="utf-8") as file:
        return json.load(file)

names = load_name_list()

versions = {}
class Module(Base):
    """
    Module is a collection of one or more modules and operations
    """

    def __init__(self, name=None):
        global versions
        global names
        super().__init__()
        self.children = []
        self.keras_operation = None
        self.sess = None
        self.predecessor = None
        self.fitness = 0
        self.logs = []
        self.db_ref = None
        self.model_image_path = None
        self.model_image_link = None

        # Identity and version-control:
        if not names:
            names = load_name_list()

        self.name = random_sample_remove(names) if not name else name

        if self.name in versions:
            self.version_number = versions[self.name]
        else:
            versions[self.name] = 0
            self.version_number = versions[self.name]
        versions[self.name] += 1

        self.ID = "{} v{}".format(self.name, self.version_number)

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity on module level. """
        from copy import deepcopy
        global versions
        new_mod = Module(self.name)
        new_mod.nodeID = self.nodeID
        new_mod.version_number = versions[self.name]

        new_mod.logs = deepcopy(self.logs)
        new_mod.ID = "{} v{}".format(new_mod.name, new_mod.version_number)

        new_mod.predecessor = self
        new_mod.children += [deepcopy(child) for child in self.children]

        # Copying connectivity for all children:
        for i, child in enumerate(self.children):
            try:
                for cn in child.next:
                    new_mod.children[i].next += [new_mod.children[self.children.index(cn)]]
                for cp in child.prev:
                    new_mod.children[i].prev += [new_mod.children[self.children.index(cp)]]

            except AttributeError as e:
                raise e
        return new_mod

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