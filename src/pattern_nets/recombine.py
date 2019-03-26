import copy
import random

from src.buildingblocks.module import Module
from src.buildingblocks.pattern import Pattern
from src.helpers import randomized_index


def get_connections_between(island_c, island_n):
    # Assign which island connects to which

    # Single option on both:
    if len(island_c) == 1 and len(island_n) == 1:
        connections = [(0, 0)]

    # Single option for one of the islands:
    elif len(island_c) > 1 and len(island_n) == 1:
        connections = [(i, 0) for i in range(len(island_c))]
    elif len(island_n) > 1 and len(island_c) == 1:
        connections = [(0, i) for i in range(len(island_n))]

    # Multiple options per island:
    else:
        reverse = len(island_c) > len(island_n)
        connections = []
        selectable = randomized_index(island_n if reverse else island_c)
        for c in (range(len(island_c)) if reverse else range(len(island_n))):
            if len(selectable) == 0:
                selectable = randomized_index(island_n if reverse else island_c)
            n, selectable = selectable[0], selectable[1:]
            connections += [(c, n) if reverse else (n, c)]

    return connections


def combine(patterns, num_nets, min_size, max_size):
    nets = []
    for i in range(num_nets):
        # Setup:
        net = Module()
        draw = randomized_index(patterns)

        for _ in range(random.randint(min_size, max_size)):
            # Selecting random patterns:
            pattern, draw = patterns[draw[0]], draw[1:]

            # Adding to net:
            net.children += [copy.deepcopy(pattern)]
            if len(draw) == 0:
                break

        # Placing 2D layers first:
        net.children.sort(key=lambda x: 0 if x.type == "2D" else 1)

        # Connecting patterns together:
        ops = []
        for i in range(1, len(net.children)):
            # Getting nets sequentially:
            x = net.children[i - 1]  # type: Pattern
            y = net.children[i]      # type: Pattern

            # Connect x and y by taking ends of x and beginnings
            # of y and creating connections:
            last = x.find_last()     # type: [Pattern]
            first = y.find_firsts()  # type: [Pattern]

            # Finding what last connects to what first:
            connections = get_connections_between(last, first)  # type: [(int, int)]

            # Applying connections:
            for xx, yy in connections:
                last[xx].next.append(first[yy])
                first[yy].prev.append(last[xx])

            # New children:
            ops += x.children
        net.children = ops + y.children
        # Done
        nets += [net]

    return nets