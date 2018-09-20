import copy
from old import template_engine
import old.frameworks.keras_compatability as ML_framework
import random


def legal_operations(operations: dict, op_names: list, previous: dict, next:dict=None) -> list:
    """
    Filters the list of all components so it only includes possible components to
    use as the next operation in the network.

    :param operations: dict of all operations.
    :param op_names: list of all operation names
    :param previous: spec for previous operation
    :param last: is last layer?
    :return: list of the names of all legal operations to use next.
    """
    if previous:
        legal = [op for op in op_names if op in previous["possibleNext"]]
    else:
        legal = [op for op in op_names if operations[op]["initialCompatible"]]

    if next:
        legal = [op for op in legal if next["name"] in operations[op]["possibleNext"]]

    return legal


def generate_random_module(previous: dict, operations: dict, op_names: list, op_count: int) -> dict:
    """
    Generates a random module with some operations.
    :param operations: dict of all operations
    :param op_names: list of all operation names
    :param op_count: number of operations this module should have
    :return: complete module
    """

    module = {"components": [], "type": "module"}

    for o in range(op_count):
        # Adding operation to module:
        operation = add_random_operation(operations, op_names, previous)
        module["components"] += [operation]
        previous = operation
    return module


def add_random_operation(operations: dict, op_names: list, previous: dict) -> dict:
    # Filtering list and selecting next operation:
    legal = legal_operations(operations, op_names, previous)
    selected_operation = legal[random.randint(0, len(legal)-1)]

    # Randomizing the parameters of the operation:
    operation = copy.deepcopy(operations[selected_operation])
    operation = template_engine.shuffle_parameters(operation)

    # Adding operation to module:
    return operation


def add_operation_by_index(model: dict, index: int, operation: dict) -> dict:
    # TODO: TEST HERE!
    def place_operation(module: dict, ops_seen: int) -> dict:
        for i, component in enumerate(module["components"]):
            if component["type"] == "operation":
                ops_seen += 1

            elif component["type"] == "module":
                ops_seen = place_operation(component, ops_seen)

            if ops_seen == index:
                comps = module["components"]
                module["components"] = comps[:i] + [operation] + comps[i:]
                return ops_seen + 1

            elif ops_seen > index:
                return ops_seen
        return ops_seen

    # Running placement algorithm:
    place_operation(model, 0)
    return model


def remove_operation_by_index():
    pass


def list_modules(model: dict) -> list:
    modules = []
    for component in model["components"]:
        if component["type"] == "module":
            modules += [component] + list_modules(component)
    return modules


def list_operations(model: dict) -> list:
    operations = []
    for component in model["components"]:
        if component["type"] == "operation":
            operations += [component]
        elif component["type"] == "module":
            operations += list_operations(component)
    return operations


def make_valid(model: dict, operations: dict) -> dict:
    op_names = list(model.keys())

    def find_bridge_operation(previous: dict, operation: dict) -> dict:
        # Search for a legal operation to bridge two illegal operations:
        legal = legal_operations(operations, op_names, previous)
        for legal_op in legal:
            if operation["name"] in operations[legal_op]["possibleNext"]:
                return operations[legal_op]
        return None

    model_operations = list_operations(model)

    previous = model_operations[0]
    for index, operation in enumerate(model_operations[1:]):
        if operation["name"] not in previous["possibleNext"]:

            model = add_operation_by_index(
                model,
                index+1,
                find_bridge_operation(previous, operation)
            )

        previous = operation
    return model


def generate(inputs: tuple, outputs: int, template_folder: str = "./old/templates"):
    """
    Generates a complete network model
    :param inputs:
    :param outputs:
    :param template_folder:
    :return:
    """

    # Loading templates from templates folder and json files:
    operations = template_engine.build(template_folder)
    op_names = list(operations.keys())

    # Root node of the network:
    model = {"components": [], "type": "module"}

    previous = operations["Conv2D"] if len(inputs) > 2 else operations["LinearLayer"]
    template_engine.shuffle_parameters(previous)

    model["components"] += [previous]

    for m in range(3):
        if len(model["components"]) > 0:
            previous = list_operations(model)[-1]

        model["components"] += [
            generate_random_module(previous, operations, op_names, 3)
        ]

    return ML_framework.export(model, input_dimensions=inputs, classes=outputs)
