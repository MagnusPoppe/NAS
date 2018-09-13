import copy
import template_engine
import frameworks
import random

def legal_modules(operations: dict, op_names: list, previous: dict, last:bool) -> list:
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
    if last:
        legal = [op for op in legal if operations[op]["outputCompatible"]]
    return legal


def generate_random_module(operations: dict, op_names: list, op_count: int) -> dict:
    """
    Generates a random module with some operations.
    :param operations: dict of all operations
    :param op_names: list of all operation names
    :param op_count: number of operations this module should have
    :return: complete module
    """

    module = {"components": [], "type": "module"}

    previous = None
    for o in range(op_count):
        # Filtering list and selecting next operation:
        legal_operations = legal_modules(operations, op_names, previous, last=(o == op_count-1))
        selected_operation = legal_operations[random.randint(0, len(legal_operations))]

        # Randomizing the parameters of the operation:
        operation = copy.deepcopy(operations[selected_operation])
        operation = template_engine.shuffle_parameters(operation)

        # Adding operation to module:
        module["components"] += [operation]
        previous = operation
    return module


def generate(inputs: tuple, outputs: int, template_folder="./templates"):
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

    for m in range(3):
        model["components"] += [generate_random_module(operations, op_names, 3)]

    return frameworks.keras.export(model, input_dimensions=inputs, classes=outputs)