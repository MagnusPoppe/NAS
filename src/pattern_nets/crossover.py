import pickle

import copy

from src.buildingblocks.pattern import Pattern
from src.pattern_nets.initialization import set_random_connections
import multiprocessing as mp


def apply(patterns_pairs: [(Pattern, Pattern)]) -> [Pattern]:
    if len(patterns_pairs) < 2:
        return []

    crossed_over = [striped_crossover(*pair) for pair in patterns_pairs]
    return crossed_over
    # # Creating h5 files async:
    # args = [
    #     (pickle.dumps(child), pickle.dumps(p1), pickle.dumps(p2))
    #     for child, (p1, p2) in list(zip(crossed_over, patterns_pairs))
    # ]
    # pool = mp.Pool()
    # procs = pool.map_async(create_learned_knowledge_file, args)
    # pool.close()
    # result = procs.get()
    # return [pickle.loads(x) for x in result]


def striped_crossover(pattern1, pattern2):
    # Selecting random layers for crossover
    new_children = []
    for i in range(min(len(pattern1.children), len(pattern2.children))):
        if i % 2 == 0:
            new = copy.deepcopy(pattern1.children[i])
        else:
            new = copy.deepcopy(pattern2.children[i])
        new.set_new_id()
        new_children += [new]

    # Creating new pattern:
    pattern = Pattern(type=pattern1.type, layers=len(new_children))
    pattern.children = new_children

    # Setting connections between layers:
    set_random_connections(pattern)
    return pattern


def already_in_use(new, op):
    return any(op.ID == n.ID for n in new)


def create_learned_knowledge_file(args):
    pattern, parent1, parent2 = pickle.loads(args)
    return pickle.dumps(pattern)
