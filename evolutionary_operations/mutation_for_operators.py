import random
from copy import deepcopy

from evolutionary_operations.mutation_operators import connect, insert, remove, append, _is_before
from evolutionary_operations.weight_transfer import transfer_predecessor_weights
from helpers import random_sample, operators, operators1D, operators2D
from frameworks.keras_decoder import assemble
from modules.base import Base
from modules.convolution import Conv2D
from modules.dense import Dense, Dropout
from modules.module import Module


def _generate_votes(weights: list) -> list:
    votes = []
    for operation, weight in weights:
        votes += [operation] * weight
    return votes


OPERATOR_WEIGHTS = [
    ("append", 2),
    ("connect", 0),
    ("insert", 10),
    ("insert-between", 10),
    ("remove", 60),
    ("identity", 20)
]

votes = _generate_votes(OPERATOR_WEIGHTS)


def is2D(op):
    return isinstance(op, Conv2D)


def is1D(op):
    return isinstance(op, Dense) or isinstance(op, Dropout)


def get_possible_insertion_points(module:Module, operation:Base) -> tuple():

    insertion_after = []
    insertion_before = []
    first = module.find_first()
    last = module.find_last()[0]
    for child in module.children:
        if is1D(operation):
            if not is2D(child) and child != first:  # Cannot be inserted before 2D op.
                insertion_before += [child]
            if child != last:
                insertion_after += [child]  # Can be inserted after any.
        elif is2D(operation):
            if not is1D(child) and child != last: # Cannot be inserted after a linear layer
                insertion_after += [child]
            if child != first:
                insertion_before += [child] # Can be inserted before any.

        # Can only insert after input layer if shape of input is same as shape of first layer.

    return insertion_after, insertion_before


def select_operator(module):
    """ Selects what mutation operator to use """
    if len(module.children) <= 3:
        return "append"
    return random_sample(votes)


def find_placement(module, operation) -> (Module, Module):
    after, before = get_possible_insertion_points(module, operation)
    if before and after:
        # Selecting first and removing from possible lasts:
        first = random_sample(after)
        before = [node for node in before if node != first and not _is_before(first, node)]
        last = random_sample(before)
        return first, last
    return None, None

def apply_mutation_operator(module, operator, operators):
    if operator == "append":
        last = module.find_last()
        operation = random_sample(operators1D)() if is1D(last) else random_sample(operators2D)()
        module = append(module, operation)

    elif operator == "remove":
        module = remove(module, random_sample(module.children))

    elif operator == "insert" or operator == "insert-between":
        operation = random_sample(operators)()
        first, last = find_placement(module, operation)
        i = 0
        while not first or not last:
            operation = random_sample(operators)()
            first, last = find_placement(module, operation)
            i += 1
            if i == 20: break
        if first and last:
            module = insert(module, first, last, operation, operator == "insert-between")
    elif operator == "connect":
        possibilities = list(range(len(module.children)))
        module = connect(
            module=module,
            first=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))],
            last=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))]
        )
    # Else: operator == "identity": do nothing...

    return module


def mutate(module: Module, in_shape: tuple, classes: int, compilation: bool = True) -> Module:
    # Copying module to do non-destructive changes.
    mutated = deepcopy(module) if compilation else module
    mutated = apply_mutation_operator(mutated, select_operator(mutated), operators)

    # Compiles keras model from module (not done for multiple mutations):
    if compilation:
        mutated.keras_tensor = assemble(mutated, in_shape, classes, is_root=True)
        if module.predecessor:
            mutated = transfer_predecessor_weights(mutated, in_shape, classes)
    return mutated
