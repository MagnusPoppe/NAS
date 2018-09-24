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
            if len(self.children) < 1:
                self.children += [other]
            else:
                previous = self.children[-1]
                previous.next += [other]
                other.prev += [previous]
                self.children += [other]
        return self

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

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

    def find_first(self):
        def on(operation):
            if operation.prev: return on(operation.prev[0])
            return operation
        return on(self.children[0])
