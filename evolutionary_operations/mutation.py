import random
from copy import deepcopy

from helpers import random_sample, operators1D
from frameworks.keras_decoder import assemble
from modules.module import Module

# TODO: Som i resnet, legg til en skip connection.
# Vi vil at en node kan muteres til å bli koblet til en annen node uten
# at det lages en ekstra node i mellom...

def mutate(module:Module, compilation=True, compile_parameters =((784,), 10), make_copy=True, modules=None) -> Module:

    mutated = deepcopy(module) if make_copy else module

    # Selecting what module to mutate in:
    if random.uniform(0,1) < 0.5 or not modules:
        op = random_sample(operators1D)()
    else:
        op = deepcopy(random_sample(modules))

    # Selecting where to place operator:
    selected = random.uniform(0,1)

    if selected < 0.3 or len(module.children) <= 3:
        mutated.append(op)

    elif selected < 0.6:
        children = list(range(0, len(mutated.children))) # uten tilbakelegging
        mutated.insert(
            first_node=mutated.children[children.pop(random.randint(0, len(children)-1))],
            second_node=mutated.children[children.pop(random.randint(0, len(children)-1))],
            operation=op
        )

    elif selected < 1.0:
        mutated.remove(random_sample(mutated.children))

    # Compiles keras model from module:
    if compilation:
        if random.uniform(0, 1) < 0.2 or not module.predecessor:
            mutated.keras_tensor = assemble(mutated, *compile_parameters, is_root=True)
        else:
            mutated = transfer_predecessor_weights(mutated, *compile_parameters)
            print("--> {} got its weights transferred from predecessor".format(module.ID))
    return mutated

def transfer_predecessor_weights(module: Module, in_shape: tuple, classes: int):
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError. (anomaly)
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """
    predecessor = module.predecessor
    module.keras_tensor = assemble(module, in_shape, classes)
    for child in predecessor.children:
        for c in module.children:
            if c.ID == child.ID:
                try:
                    c.keras_operation.set_weights(child.keras_operation.get_weights())
                except ValueError:
                    print("    - Incompatible weights.")
                    break
    return module