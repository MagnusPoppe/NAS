from tensorflow import keras
from modules.base import Base
from modules.dense import Dropout
from modules.operation import Operation

global_id = 1


class Module(Base):
    """
    Module is a collection of one or more modules and operations
    """

    ID = ""  # Should be set random by the app.

    def __init__(self):
        super().__init__()
        self.children = []

    def __iadd__(self, other):
        if isinstance(other, Operation) or isinstance(other, Module):
            self.append(other)
        return self

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

    def append(self, op):
        if len(self.children) < 1:
            self.children += [op]
        else:
            previous = self.children[-1]
            previous.next += [op]
            op.prev += [previous]
            self.children += [op]

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

    def compile(self, input_shape, classes):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :return: tf.keras.model.Model
        """

        # TODO: Parse the whole graph to connect all ends.

        def compute_graph(current: Operation):
            # Edge case, first node in network:
            if len(current.prev) == 0:
                operation = current.to_keras()(current.input)

            # Normal sequential add:
            elif len(current.prev) == 1:
                operation = current.to_keras()(current.prev[0].keras_operation)

            # More than one input, need to merge:
            else:
                if all(not op.keras_operation is None for op in current.prev):
                    concat = keras.layers.concatenate([op.keras_operation for op in current.prev])
                    operation = current.to_keras()(concat)
                else:
                    operation = current.to_keras()

            current.keras_operation = operation
            last_layer = current

            # Special case: If a merge happens, only continue when all earlier branches has finished.
            if all(not op.keras_operation is None for op in current.prev):
                for op in current.next:
                    last_layer = compute_graph(op)

            return last_layer

        input = keras.layers.Input(shape=input_shape)
        first_node = self.find_first()
        first_node.input = input
        last_op = compute_graph(first_node)

        output = keras.layers.Dense(units=classes, activation="softmax")(last_op.keras_operation)
        return keras.models.Model(inputs=[input], outputs=[output])