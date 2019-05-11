import json, os

from src.helpers import random_sample_remove
from src.buildingblocks.base import Base
from src.pattern_nets.pattern_connector import get_connections_between

global_id = 1

def load_name_list():
    with open("./resources/names_clean.json", "r", encoding="utf-8") as file:
        return json.load(file)


names = load_name_list()
versions = {}


def reset_naming():
    global versions
    global names
    names = load_name_list()
    versions = {}


def get_name_and_version(name: str) -> (str, int):
    global versions
    global names
    if len(names) == 0:
        reset_naming()

    name = name if name else random_sample_remove(names)

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
        self.immediate_transferred_knowledge_epochs = 0
        self.transferred_knowledge_epochs = 0

    def __str__(self):
        return "{} [{}]".format(self.ID, ", ".join([str(c) for c in self.children]))

    def __deepcopy__(self, memodict={}, clone=None):
        """ Does not retain connectivity on module level. """
        from copy import deepcopy

        # Core values:
        new_mod = clone if clone else Module(self.name)
        new_mod.logs = deepcopy(self.logs)

        # Identity and version-control:
        new_mod.name, new_mod.version = get_name_and_version(self.name)
        new_mod.ID = "{} v{}".format(new_mod.name, new_mod.version)
        new_mod.immediate_transferred_knowledge_epochs = self.epochs_trained
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

    def acc(self):
        try:
            return self.fitness[-1]
        except IndexError:
            return 0.0

    def val_acc(self):
        try:
            return self.validation_fitness[-1]
        except IndexError:
            return 0.0

    def test_acc(self):
        try:
            return self.latest_report()['weighted avg']["precision"]
        except (KeyError, IndexError):
            return 0.0

    def latest_report(self):
        keys = list(self.report.keys())
        keys.sort()
        return self.report[keys[-1]]

    def get_improvement(self):
        if len(self.report) >= 1 and self.predecessor and len(self.predecessor.report) >= 1:
            acc = self.test_acc()
            pred_acc = self.predecessor.test_acc()
            impr = acc - pred_acc
            return impr
        else:
            return 0.0

    def get_session_improvement(self):
        if len(self.report) > 1:
            keys = list(self.report.keys())
            keys.sort()
            return self.report[keys[-1]]['weighted avg']["precision"] - self.report[keys[-2]]['weighted avg']["precision"]
        elif len(self.report) == 1 and self.predecessor:
            return self.get_improvement()
        else:
            return self.test_acc()
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

    def find_firsts(self) -> [Base]:
        """ Finds all first nodes in module directional graph of operations """
        def find_start(comp, seen):
            starts = []
            if comp in seen:
                return starts
            seen += [comp]
            if comp.prev:
                for _prev in comp.prev:
                    starts += find_start(_prev, seen)
            else:
                starts += [comp]
            return starts

        start = []
        seen = []
        for child in self.children:
            start += find_start(child, seen)
        return start

    def find_last(self) -> [Base]:
        """ Finds all ends in the directional graph of operations """
        def find_end(comp:Base, seen) -> list:
            ends = []
            if comp in seen:
                return ends

            seen += [comp]
            if comp.next:
                for next_module in comp.next:
                    ends += find_end(next_module, seen)
            else:
                ends += [comp]
            return ends

        ends = []
        seen = []
        for child in self.children:
            ends += find_end(child, seen)
        return ends

    def absolute_save_path(self, config):
        """ Reveals the absolute path to storage directory """
        return config.results.ensure_individ_path(self)

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

    def contains_duplicates(self):
        return any([
            any([
                cx.ID == cy.ID and i != j
                for j, cx in enumerate(self.children)
            ])
            for i, cy in enumerate(self.children)
        ])

    def model_file_exists(self, config):
        path = os.path.join(self.absolute_save_path(config), "model.h5")
        if os.path.isfile(path):
            self.saved_model = path
            return True
        return False

    def connect_all_sub_modules_sequential(self):
        ops = []
        if len(self.children) == 1:
            ops = self.children[0].children
        else:
            for i in range(1, len(self.children)):
                # Getting nets sequentially:
                x = self.children[i - 1]  # type: Pattern
                y = self.children[i]  # type: Pattern

                # Connect x and y by taking ends of x and beginnings
                # of y and creating connections:
                last = x.find_last()  # type: [Pattern]
                first = y.find_firsts()  # type: [Pattern]

                # Finding what last connects to what first:
                connections = get_connections_between(last, first)  # type: [(int, int)]

                # Applying connections:
                for xx, yy in connections:
                    last[xx].next += [first[yy]]
                    first[yy].prev += [last[xx]]

                # New children:
                ops += x.children
                if i == len(self.children) - 1:
                    ops += y.children
        return ops

