import random

import copy

from src.buildingblocks.ops.operation import Operation
from src.helpers import random_sample, random_sample_remove, operators2D_votes, operators1D_votes


def change_connection(pattern):
    return pattern


def remove_op(pattern):
    removable = random_sample_remove(pattern.children)  # type: Operation
    removable.disconnect()
    return pattern


def swap_op(pattern):
    new_op_class = random_sample(operators2D_votes) \
        if pattern.type == "2D" \
        else random_sample(operators1D_votes)

    new_op = new_op_class()
    old_op = random_sample_remove(pattern.children)  # type: Operation
    new_op.inherit_connectivity_from(old_op)
    pattern.children += [new_op]
    return pattern


def apply(patterns):
    mutated = []
    for i, pattern in enumerate(patterns):
        if len(pattern.children) <= 2:
            operators = [swap_op]
        else:
            operators = [remove_op]  # , change_connection ]

        p = copy.deepcopy(pattern)
        for op in p.children:
            op.set_new_id()
        operator = operators[i % len(operators)]
        p = operator(p)
        mutated += [p]
    return mutated
