from tensorflow import keras
from modules.base import Base
from modules.dense import Dropout
from modules.operation import Operation
import networkx as nx

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
        import matplotlib.pyplot as plt
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

    def compile(self, input_shape):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :return: tf.keras.model.Sequential
        """
        # TODO: Parse the whole graph to connect all ends.

        def compute_graph(current: Operation, model: keras.models.Sequential):
            inn_shape = None

            # Merge case, only if all previous models has been completed:
            if len(current.prev) > 1 and all([x.model != None for x in current.prev]):
                outputs = [op.model.output for op in current.prev]
                concat = keras.layers.concatenate(outputs)
                inn_shape = concat.shape

                model.add(keras.layers.Dense(units=inn_shape[1].value)) # TODO: Only works with linear layers.

            # Split case:
            if len(current.next) > 1:
                if not inn_shape:
                    inn_shape = input_shape if not current.prev else current.find_shape()
                for op in current.next[1:]:
                    split_model = keras.models.Sequential()
                    split_model.add(model)
                    compute_graph(op, split_model)

            # Adding current node into compute graph:
            operation = current.to_keras()
            model.add(operation)
            current.model = model
            if current.next and (len(current.prev) <= 1 or all([x.model != None for x in current.prev])):
                compute_graph(current.next[0], model)

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape))
        compute_graph(self.find_first(), model)
        return model

    def find_first(self):
        def on(operation):
            if operation.prev:
                for p in operation.prev:
                    return on(p)
            return operation
        return on(self.children[0])

    def get_ends(self):
        start = self.find_first()

