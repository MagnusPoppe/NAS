from tensorflow import keras
from modules.base import Base
from modules.operation import Operation


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
        pass

    def compile(self, input_shape):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :return: tf.keras.model.Sequential
        """

        # TODO: Parse the whole graph to connect all ends.

        def compute_graph(current: Operation, model: keras.models.Sequential):
            shape = (current.prev[0].shape) if len(current.prev) == 1 else input_shape

            # Merge case, only if all previous models has been completed:
            if len(current.prev) > 1 and all([x.model != None for x in current.prev]):
                concat = keras.layers.Concatenate([x.model for x in current.prev])
                shape = concat.output_shape
                model.add(concat)


            # Split case:
            if len(current.next) > 1:
                for op in current.next[1:]:
                    split_model = keras.models.Sequential()
                    split_model.add(keras.layers.InputLayer(shape))
                    compute_graph(op, split_model)

            # Adding current node into compute graph:
            operation = current.to_keras()
            model.add(operation)
            current.model = model
            current.shape = operation.output_shape
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
