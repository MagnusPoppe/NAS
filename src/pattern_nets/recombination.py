import copy

import numpy as np

from src.buildingblocks.module import Module
from src.buildingblocks.pattern import Pattern
from src.pattern_nets import connecter


def draw_almost_random_net(nets, max_size, avoid=None):
    nets = [net for net in nets if not avoid in net.children and len(net.children) <= max_size]
    if not nets:
        return None

    alpha = sum([len(net.children) for net in nets])
    if alpha == 0:
        return nets[np.random.randint(0, len(nets))]
    probabilities = [1 - (len(net.children) / alpha) for net in nets]
    distribution = []
    for i, net in enumerate(nets):
        distribution += [net] * int(probabilities[i] * 100)
    return distribution[np.random.randint(0, len(distribution))] if distribution else None


def combine_random(patterns: [Pattern], num_nets: int, max_size: int, redraw: bool = False) -> [Module]:
    """
    Combines patterns to nets. Constraints:
    - All patterns must be included in at least 1 net.
    - A pattern may not be repeated in a net.

    :param patterns: The pool of patterns
    :param num_nets: Count of how many nets to be generated,
    :param max_size: How many patterns maximum per network?
    :param redraw: reuses operations if random and if can (given constrains).
    :return: List of generated nets with length equal to output_nets
    """
    nets = [Module() for _ in range(num_nets)]
    draw = connecter.randomized_index(patterns)

    # Assigning patterns to nets:
    while len(draw) > 0 or redraw:
        index, draw = draw[0], draw[1:]
        pattern = copy.deepcopy(patterns[index])
        net = draw_almost_random_net(nets, max_size=max_size)
        if not net:
            break

        net.children += [pattern]

        # Consider restart:
        if redraw and len(draw) == 0:
            # Checking if any patterns can be used:
            for pattern in patterns:
                can_use_pattern = all([draw_almost_random_net(net, max_size, pattern) is None for net in nets])
                if not can_use_pattern:
                    pattern.pop(pattern.index(pattern))

            # Will restart if available patterns and with 50% chance:
            if patterns and np.random.randint(0, 1) == 1:
                draw = connecter.randomized_index(patterns, np.random.randint(0, len(patterns)))
            else:
                break

    for net in nets:
        type_2D = [child for child in net.children if child.type == "2D"]
        type_1D = [child for child in net.children if child.type == "1D"]
        connecter.island_join(type_2D + type_1D)
    return nets


def combine_optimal(patterns: [Pattern]) -> Module:
    pass
