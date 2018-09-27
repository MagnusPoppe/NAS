from copy import copy

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

    def __copy__(self):
        new_mod = Module()
        new_mod.ID = self.ID + "_copy"
        new_mod.nodeID = self.nodeID
        for child in self.children:
            new_mod.children += [copy(child)]
        for i, child in enumerate(self.children):
            nexts = [self.children.index(cn) for cn in child.next]
            prevs = [self.children.index(cp) for cp in child.prev]
            new_mod.children[i].next += [new_mod.children[n] for n in nexts]
            new_mod.children[i].prev += [new_mod.children[p] for p in prevs]
        return new_mod

    def append(self, op):
        if len(self.children) == 0:
            self.children += [op]
        elif len(self.children) == 1:
            self.children[0].next += [op]
            op.prev += [self.children[0]]
            self.children += [op]
        else:
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

    def to_keras(self):
        return self.compile(None)

    def decompile(self):
        def remove_keras_ops(module):
            module.keras_operation = None
            module.keras_tensor = None
            try: module.input = None
            except AttributeError: pass

            for child in module.children:
                if isinstance(child, Operation):
                    try: module.input = None
                    except AttributeError: pass
                    module.keras_operation = None
                    module.keras_tensor = None
                elif isinstance(child, Module):
                    remove_keras_ops(child)
        remove_keras_ops(self)


    def compile(self, input_shape, classes=0, is_root=False):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :param input: shape tuple
        :return: tf.keras.model.Model
        """

        def _connect(current, previous_tensor):
            if isinstance(current, Module):
                if "input" in previous_tensor.name:
                    current.keras_operation = current.compile(previous_tensor, is_root=True)
                else:
                    current.keras_operation = current.compile(tuple(previous_tensor.shape), is_root=True)
            else: # must be of type: Operation
                current.keras_operation = current.to_keras()
            current.keras_tensor = current.keras_operation(previous_tensor)
            return current

        self.decompile()

        queue = [self.find_first()]
        input = keras.layers.Input(shape=input_shape) if isinstance(input_shape, tuple) else input_shape
        queue[0].input = input
        ends = []
        processed = []

        while len(queue) > 0:
            current = queue.pop(0)
            if current in processed: continue # Edge case. Nodes may be queued multiple times.

            # Trying to find the previous output tensor:
            if len(current.prev) == 0:
                # First node has previous tensor as input layer
                prev = input
            elif len(current.prev) == 1 and current.prev[0].keras_operation is not None:
                # Only one previous node:
                prev = current.prev[0].keras_tensor
            elif len(current.prev) >= 2 and all(not op.keras_operation is None for op in current.prev):
                # Multiple previous nodes. Needs Concatinate layer.
                # NOTE: Requires all previous operations to be created.
                prev = keras.layers.concatenate([op.keras_tensor for op in current.prev])
            else:
                # Is not ready. Add back in queue...
                queue.append(current); continue
            processed += [current]

            # Connecting node to previous layer:
            current = _connect(current, prev) # --> recursive call if current is type Module

            if current.next: queue += [n for n in current.next]
            else: ends += [current]

        # Handling multiple ends for the network:
        if len(set(ends)) > 1: end = keras.layers.concatenate([op.keras_tensor for op in set(ends)])
        else: end = ends[0].keras_tensor

        out =  keras.layers.Dense(units=classes, activation="softmax")(end) if not is_root else end
        try:
            self.keras_operation = keras.models.Model(inputs=[input],outputs=[out])
        except ValueError as e:
            raise e
        self.keras_tensor = self.keras_operation.layers[-1].output
        return self.keras_operation

    def remove(self, child: Base):
        # Removing from list:
        index = self.children.index(child)
        self.children.pop(index)

        # Cutting ties:
        for p in child.next: p.prev.remove(child)
        for p in child.prev: p.next.remove(child)



