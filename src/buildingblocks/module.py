import json, os

from src.helpers import random_sample_remove
from src.buildingblocks.base import Base

global_id = 1

def load_name_list():
    with open("./resources/names_clean.json", "r", encoding="utf-8") as file:
        return json.load(file)


names = load_name_list()
versions = {}


def get_name_and_version(name: str) -> (str, int):
    global versions
    global names
    name = random_sample_remove(names) if not name else name

    if name in versions:
        version = versions[name]
    else:
        versions[name] = 0
        version = versions[name]
    versions[name] += 1
    return name, version


class Module(Base):
    """
    Module is a collection of one or more modules and operations that form
    a neural network
    """

    def __init__(self, name=None):
        super().__init__()

        # Core values:
        self.children = []
        self.predecessor = None
        self.re_train = False
        self.logs = []

        # Identity and version-control:
        self.name, self.version = get_name_and_version(name)
        self.ID = "{} v{}".format(self.name, self.version)

        # Metrics:
        self.fitness = []
        self.evaluation = {}
        self.loss = []
        self.validation_fitness = []
        self.validation_loss = []
        self.report = {}

        # Database:
        self.db_ref = None
        self.model_image_path = None
        self.model_image_link = None
        self.saved_model = None
        self.epochs_trained = 0
        self.transferred_knowledge_epochs = 0

    def __str__(self):
        return "{} [{}]".format(self.ID, ", ".join([str(c) for c in self.children]))

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity on module level. """
        from copy import deepcopy

        # Core values:
        new_mod = Module(self.name)
        new_mod.logs = deepcopy(self.logs)

        # Identity and version-control:
        new_mod.name, new_mod.version = get_name_and_version(self.name)
        new_mod.ID = "{} v{}".format(new_mod.name, new_mod.version)
        new_mod.transferred_knowledge_epochs = self.transferred_knowledge_epochs + self.epochs_trained

        # Connectivity:
        new_mod.predecessor = self
        new_mod.children += [deepcopy(child) for child in self.children]
        for i, child in enumerate(self.children):
            for cn in child.next:
                new_mod.children[i].next += [new_mod.children[self.children.index(cn)]]
            for cp in child.prev:
                new_mod.children[i].prev += [new_mod.children[self.children.index(cp)]]
        return new_mod

    def number_of_operations(self) -> int:
        """ Calculates how many operations are in this Module.
            Including operations of sub-modules
        """
        modules = [m for m in self.children if isinstance(m, Module)]
        return (len(self.children) - len(modules)) + sum(m.number_of_operations() for m in modules)

    def find_first(self) -> Base:
        """ Find first node in module directional graph of operations """
        def on(operation):
            if operation.prev: return on(operation.prev[0])
            return operation
        return on(self.children[0])

    def find_last(self) -> [Base]:
        """ Finds all ends in the directional graph of operations """
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

    def relative_save_path(self, config):
        """ Reveals the storage directory relative to work directory """
        path = 'results/{}/{}/v{}'.format(config.results_name, self.name, self.version)
        os.makedirs(path, exist_ok=True)
        return path

    def absolute_save_path(self, config):
        """ Reveals the absolute path to the storage directory"""
        return os.path.abspath(self.relative_save_path(config))

    def clean(self):
        """ Removes all traces of keras from the module.
            This is needed to serialize object.
        """
        def detach_keras(obj):
            try: del obj.keras_operation
            except AttributeError: pass
            try: del obj.keras_tensor
            except AttributeError: pass
            try: del obj.layer
            except AttributeError: pass
            try: del obj.tensor
            except AttributeError: pass

        detach_keras(self)
        for child in self.children:
            if isinstance(child, Module):
                child.clean()
            else:
                detach_keras(child)
