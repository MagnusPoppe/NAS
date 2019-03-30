import copy

from src.buildingblocks.pattern import Pattern
from src.helpers import random_sample
from src.pattern_nets.initialization import set_random_connections


def apply(patterns_pairs: [(Pattern, Pattern)]) -> [Pattern]:
    crossed_over = [striped_crossover(*pair) for pair in patterns_pairs]

    # TODO: Create new h5 file with new weights for transfer learning

    return crossed_over


def striped_crossover(pattern1, pattern2):
    # Selecting random layers for crossover
    new_children = []
    for i in range(min(len(pattern1.children), len(pattern2.children))):
        if i % 2 == 0:
            new_children += [copy.deepcopy(pattern1.children[i])]
        else:
            new_children += [copy.deepcopy(pattern2.children[i])]

    # Creating new pattern:
    pattern = Pattern(type=pattern1.type, layers=len(new_children))
    pattern.children = new_children

    # Setting connections between layers:
    set_random_connections(pattern)
    return pattern
