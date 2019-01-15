from operator import attrgetter

import numpy as np


def tournament(population: list, individs_to_select: int):
    """
    Selection by tournament. Creates two randomized lists of
    individ ids. Selection is done by drawing two and two out
    of the randomized list and selecting the best of the two.
    :param population:
    :param individs_to_select:
    :return: Generator
    """
    individs = np.array(range(len(population)))
    np.random.shuffle(individs)
    individs = individs[:individs_to_select]
    selector = zip(
        individs[:int(len(individs) / 2)],
        individs[int(len(individs) / 2):]
    )

    for i, j in selector:
        yield population[i] if (population[i].fitness > population[j].fitness) \
            else population[j]


def trash_bad_modules(modules: list, evaluate, modules_to_keep: int = 20) -> list:
    """ Removes the worst modules from the global list of modules.
        This is for memory optimization. A lot of memory is used by
        each decoded module, so for computers with little RAM, having
        a big list of modules will slow the system down.

        NOTE: This number should be higher than the population size.
    """
    if len(modules) < modules_to_keep:
        return modules
    # evaluate(modules)
    modules.sort(key=attrgetter("fitness"))
    print("--> Deleted {} modules".format(len(modules) - modules_to_keep))
    return modules[len(modules) - modules_to_keep:]
