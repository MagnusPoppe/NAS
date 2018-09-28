import random
from copy import deepcopy

from helpers import random_sample, operators1D
from module_decoder import assemble
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
        mutated.keras_tensor = assemble(mutated, *compile_parameters, is_root=True)
    return mutated