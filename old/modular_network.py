import random

from old.network import legal_operations
from old import template_engine
import old.frameworks.keras_compatability as ML_framework
import copy


def _define_input(all_operations, inputs):
    op = "InputDense"
    if len(inputs) > 1:
        op = "InputND"

    input_op = copy.deepcopy(all_operations[op])
    input_op["parameters"]["shape"] = list(inputs)
    return input_op


def new_module():
    return {"ops": [], "prev": [], "next": [], "type": "module"}


def add_random_operation(module: dict, templates:dict) -> dict:
    def recursive_add(mod):
        # Module of modules:
        if mod["ops"] and mod["ops"][0]["type"] == "module":
            mod = recursive_add(mod["ops"][random.randint(0, len(mod["ops"]) - 1)])

        # Module of operations:
        elif mod["ops"] and mod["ops"][0]["type"] == "operation":
            # Finding previous and next operation:
            selectable = list(range(len(mod["ops"])))
            prev = selectable.pop(random.randint(0, len(selectable) - 1))
            next = random.randint(1, len(selectable) - 1) if len(selectable) > 1 else None

            # Selecting operation:
            if next or next == 0:
                legal_ops = legal_operations(templates, list(templates.keys()), mod["ops"][prev], mod["ops"][next])
            else:
                legal_ops = legal_operations(templates, list(templates.keys()), mod["ops"][prev])

            if legal_ops:
                op = copy.deepcopy(templates[legal_ops[random.randint(0, len(legal_ops) - 1)]])
                # TODO: Shuffle params

                # Connecting together:
                mod["ops"][prev]["next"] += [op]
                op["prev"] += [mod["ops"][prev]]
                if (next or next == 0) and mod["ops"][next]["prev"]:
                    mod["ops"][next]["prev"] += [op]
                    op["next"] += [mod["ops"][next]]

                mod["ops"] += [op]

        # Empty module:
        else:
            legal_ops = legal_operations(templates, list(templates.keys()), None)
            mod["ops"] += [templates[legal_ops[random.randint(0, len(legal_ops) - 1)]]]
        return mod

    return recursive_add(module)




def generate(inputs: tuple, outputs: int, template_folder: str = "./templates") -> dict:
    """
    Generates a complete network model
    :param inputs:
    :param outputs:
    :param template_folder:
    :return:
    """

    # Loading templates from templates folder and json files:
    templates = template_engine.build(template_folder)

    # Root node of the network:
    model = new_module()
    model = add_random_operation(model, templates)
    model = add_random_operation(model, templates)
    model = add_random_operation(model, templates)
    model = add_random_operation(model, templates)
    model = add_random_operation(model, templates)

    root = new_module()
    root["ops"] += [model]
    ML_framework.assemble(model)

    return model


if __name__ == '__main__':
    # FOR DEBUGGER ONLY:
    inputs = (28, 28, 1)
    outputs = 10
    model = generate(inputs, outputs)
    # ML_framework.export(model, input_dimensions=inputs, classes=outputs)
