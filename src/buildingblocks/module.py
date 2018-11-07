import json, os

from src.helpers import random_sample_remove
from src.buildingblocks.base import Base

global_id = 1

def load_name_list():
    with open("./resources/names_clean.json", "r", encoding="utf-8") as file:
        return json.load(file)


names = load_name_list()
versions = {}


class Module(Base):
    """
    Module is a collection of one or more modules and operations that form
    a neural network
    """

    def __init__(self, name=None):
        global versions
        global names
        super().__init__()
        self.children = []
        self.predecessor = None
        self.fitness = 0
        self.logs = []
        self.db_ref = None
        self.model_image_path = None
        self.model_image_link = None
        self.saved_model = None
        self.epochs_trained = 0

        # Identity and version-control:
        self.name = name if name else random_sample_remove(names)

        if self.name in versions:
            self.version = versions[self.name]
        else:
            versions[self.name] = 0
            self.version = versions[self.name]
        versions[self.name] += 1

        self.ID = "{} v{}".format(self.name, self.version)

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity on module level. """
        from copy import deepcopy
        global versions
        new_mod = Module(self.name)
        new_mod.nodeID = self.nodeID
        new_mod.version = versions[self.name]

        new_mod.logs = deepcopy(self.logs)
        new_mod.ID = "{} v{}".format(new_mod.name, new_mod.version)

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

    def number_of_operations(self):
        modules = [m for m in self.children if isinstance(m, Module)]
        return (len(self.children) - len(modules)) + sum(m.number_of_operations() for m in modules)

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

    def get_relative_module_save_path(self, config):
        path = 'results/{}/{}/v{}'.format(config['run id'], self.name, self.version)
        os.makedirs(path, exist_ok=True)
        return path

    def get_absolute_module_save_path(self, config):
        return os.path.abspath(self.get_relative_module_save_path(config))

    # def visualize(self):
    #     # Local imports. Server does not have TKinter and will crash on load.
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #
    #     G = nx.DiGraph()
    #
    #     def draw(prev, current):
    #         if current.nodeID is None:
    #             global global_id
    #             current.nodeID = "{}: {}".format(global_id, current.ID)
    #             global_id += 1
    #
    #         if prev:
    #             G.add_node(current.nodeID)
    #             G.add_edge(prev.nodeID, current.nodeID)
    #         else:
    #             G.add_node(current.nodeID)
    #
    #         if len(current.prev) <= 1 or all([x.nodeID != None for x in current.prev]):
    #             for node in current.next:
    #                 draw(current, node)
    #
    #     draw(prev=[], current=self.find_first())
    #
    #     plt.subplot(111)
    #     nx.draw(G, with_labels=True, arrowsize=1, arrowstyle='fancy')
    #     plt.show()
