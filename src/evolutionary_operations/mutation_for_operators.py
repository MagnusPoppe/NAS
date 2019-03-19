import random
from copy import deepcopy

from src.buildingblocks.ops.pooling import Pooling
from src.evolutionary_operations.mutation_operators import (
    connect,
    insert,
    remove,
    append,
    _is_before,
)
from src.helpers import (
    random_sample,
    operators_votes,
    operators1D_votes,
    operators2D_votes,
    generate_votes,
)
from src.buildingblocks.base import Base
from src.buildingblocks.ops.convolution import Conv2D
from src.buildingblocks.ops.dense import Dense # , Dropout
from src.buildingblocks.module import Module


OPERATOR_WEIGHTS = [
    ("append", 2),
    ("connect", 0),
    ("insert", 2),
    ("insert-between", 30),
    ("remove", 30)
    # ("identity", 20)
]

votes = generate_votes(OPERATOR_WEIGHTS)


def is2D(op):
    return isinstance(op, Conv2D) or isinstance(op, Pooling)


def is1D(op):
    return isinstance(op, Dense) # or isinstance(op, Dropout)


def get_possible_insertion_points(module: Module, operation: Base) -> (list, list):
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
            if (
                not is1D(child) and child != last
            ):  # Cannot be inserted after a linear layer
                insertion_after += [child]
            if child != first:
                insertion_before += [child]  # Can be inserted before any.

        # Can only insert after input layer if shape of input is same as shape of first layer.

    return insertion_after, insertion_before


def select_operator(module: Module):
    """ Selects what mutation operator to use """
    if len(module.children) <= 3:
        return "append"
    return random_sample(votes)


def find_placement(module: Module, operation: Base) -> (Module, Module):
    after, before = get_possible_insertion_points(module, operation)
    if before and after:
        # Selecting first and removing from possible lasts:
        first = random_sample(after)
        before = [
            node for node in before if node != first and not _is_before(first, node)
        ]
        last = random_sample(before)
        return first, last
    return None, None


def apply_mutation_operator(module: Module, operator: Base, operators: list) -> Module:
    if operator == "append":
        last = module.find_last()[0]
        if is1D(last):
            operation = random_sample(operators1D_votes)()
        else:
            operation = random_sample(operators2D_votes + operators1D_votes)()
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
            if i == 20:
                break
        if first and last:
            module = insert(
                module, first, last, operation, operator == "insert-between"
            )
    elif operator == "connect":
        possibilities = list(range(len(module.children)))
        module = connect(
            module=module,
            first=module.children[
                possibilities.pop(random.randint(0, len(possibilities) - 1))
            ],
            last=module.children[
                possibilities.pop(random.randint(0, len(possibilities) - 1))
            ],
        )
    # Else: operator == "identity": do nothing...

    return module


def mutate(module: Module, make_copy: bool = True) -> Module:
    # Copying module to do non-destructive changes.
    mutated = deepcopy(module) if make_copy else module
    mutated = apply_mutation_operator(mutated, select_operator(mutated), operators_votes)
    return mutated
