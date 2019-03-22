import numpy as np

from src.buildingblocks.module import Module
from src.buildingblocks.pattern import Pattern
from src.pattern_nets.operations import connect


def randomized_index(patterns: [Pattern], index_size: int = 0) -> np.array:
    index_size = len(patterns) if index_size == 0 else index_size
    draw = np.arange(len(patterns))
    np.random.shuffle(draw)
    return draw[:index_size]


def draw_almost_random_net(nets, avoid=None):
    nets = [net for net in nets if not avoid in net.children]
    if not nets:
        return None

    alpha = sum([len(net.children) for net in nets])
    if alpha == 0:
        return nets[np.random.randint(0, len(nets))]
    probabilities = [1 - (len(net.children) / alpha) for net in nets]
    distribution = []
    for i, net in enumerate(nets):
        distribution += [net] * int(probabilities[i] * 100)
    return distribution[np.random.randint(0, len(distribution))]


def combine_random(patterns: [Pattern], num_nets: int, redraw: bool = False) -> [Module]:
    """
    Combines patterns to nets. Constraints:
    - All patterns must be included in at least 1 net.
    - A pattern may not be repeated in a net.

    :param patterns: The pool of patterns
    :param num_nets: Count of how many nets to be generated
    :param redraw: reuses operations if random and if can (given constrains).
    :return: List of generated nets with length equal to output_nets
    """
    nets = [Module() for _ in range(num_nets)]
    draw = randomized_index(patterns)

    # Assigning patterns to nets:
    while len(draw) > 0 or redraw:
        index, draw = draw[0], draw[1:]
        pattern = patterns[index]
        net = draw_almost_random_net(nets)
        net.children += [pattern]

        # Consider restart:
        if redraw and len(draw) == 0:
            for pattern in patterns:
                can_use_pattern = all([draw_almost_random_net(net, pattern) is None for net in nets])
                if not can_use_pattern:
                    pattern.pop(pattern.index(pattern))
            if patterns and np.random.randint(0, 1) == 1:
                draw = randomized_index(patterns, np.random.randint(0, len(patterns)))
            else:
                break

    for net in nets:
        net.children.sort(key=lambda x: x.type == "2D", reverse=True)
        children = []
        for i in range(len(net.children)-1):
            first = net.children[i].find_firsts()
            last = net.children[i+1].find_last()
            for f in first:
                for l in last:
                    l.next += [f]
                    f.prev += [l]
                    children += [l]
                children += [f]
        net.children = list(set(children))
    return nets


def combine_optimal(patterns: [Pattern]) -> Module:
    pass
