import tensorflow as tf
from tensorflow import keras


def export(root: dict, input_dimensions: tuple, classes:int) -> keras.Sequential:
    def find_first_layer(node):
        if node["type"] == "module" and len(node["components"]) > 0:
            return find_first_layer(node["components"][0])
        elif node["type"] == "operation":
            return node

    def parse(node, model):
        if node["type"] == "module":
            for component in node["components"]:
                parse(component, model)

        elif node["type"] == "operation":
            model.add(to_keras(node))

    first = find_first_layer(root)
    first["parameters"]["input"] = input_dimensions
    model = keras.Sequential()

    parse(root, model)
    model.add(keras.layers.Dense(classes, activation="softmax"))
    return model


def assemble(root: dict) -> dict:

    def asseble_module(module: dict) -> dict:
        # TODO: HOW TO CREATE MAD CRAZY EVOLVED STRUCTURES:
        # Create a new Sequential Model for all "splits".
        # Create serial component for all single layer transfers.
        # Create merges for where more than one previous exists.

        def gather_operations(op):
            ops = [to_keras(op)]
            for next in op["next"]:
                ops += [gather_operations(next)]
            return ops

        first = module["ops"][0]
        operations = [gather_operations(op) for op in first["next"]]
        module["model"] = keras.models.Sequential()

    return asseble_module(root)


def to_keras(operation):
    params = operation["parameters"]

    if operation["name"] == "LinearLayer":
        if "input" in params:
            return keras.layers.Dense(
                units=params["neurons"]["value"],
                activation=params["activation"]["value"],
                input_shape=params["input"]
            )
        else:
            return keras.layers.Dense(
                units=params["neurons"]["value"],
                activation=params["activation"]["value"]
            )


    elif operation["name"] == "Conv2D":
        if "input" in params:
            return keras.layers.Conv2D(
                filters= params["filters"]["value"],
                kernel_size=(params["kernelWidth"]["value"], params["kernelHeight"]["value"]),
                input_shape = params["input"]
            )
        else:
            return keras.layers.Conv2D(
                filters=params["filters"]["value"],
                kernel_size=(params["kernelWidth"]["value"], params["kernelHeight"]["value"]),
            )

    elif operation["name"] == "Dropout":
        return keras.layers.Dropout(rate=params["rate"]["value"])

    elif operation["name"] == "Flatten":
        return keras.layers.Flatten()

    operation["parameters"] = params