import sys
import functools


def fast_non_dominated_sort(solutions: list, dominates: callable):
    """
    Multi-objective optimization sort for finding the
    best solutions based on multiple objectives.

    :param dominates:
        This represents the crowded comparison operator
        p dominates q if BOTH conditions are met:
            1. p is no worse than q in all objectives
            2. p is strictly better than q on atleast one objective
    """

    fronteer = [[]]

    for solution in solutions:
        solution.dominates = []
        solution.dominated_by = 0

        # Checking for which solutions are dominated:
        for other in solutions:
            if dominates(solution, other):
                solution.dominates += [other]
            elif dominates(other, solution):
                solution.dominated_by += 1

        # If solution belongs in first front
        if solution.dominated_by == 0:
            solution.rank = 0
            fronteer[0] += [solution]

    i = 0
    while len(fronteer[i]) > 0:
        Q = []  # Stores members of the next front

        for solution in fronteer[i]:
            for other in solution.dominates:
                other.dominated_by -= 1

                # If other belongs in next front
                if other.dominated_by == 0:
                    Q += [other]
                    other.rank = i + 1
        i += 1
        fronteer += [Q]
    return fronteer


def crowding_distance_assignment(solutions: list, objectives: [callable]):
    """ Assigns distances between modules to maintain diversity. This
        function will try to reward bigger differences between
        different parameters.
    """
    # initially, all distances should be 0
    for s in solutions:
        s.distance = 0

    for objective in objectives:
        # Maximum and minimum values of objective:
        objective_min = min([objective(s) for s in solutions])
        objective_max = max([objective(s) for s in solutions])

        # Initializing values for objective:
        solutions.sort(key=objective)
        solutions[0].distance = len(solutions) * 1.0  # (sys.maxsize) Max reward per category
        solutions[-1].distance = len(solutions) * 1.0  # (sys.maxsize) Max reward per category

        # Setting distances:
        for i in range(1, len(solutions) - 1):
            solutions[i].distance += (
                objective(solutions[i + 1]) - objective(solutions[i - 1])
            ) / (objective_max - objective_min)


def nsga_ii(solutions: list, objectives: [callable], domination_operator: callable):
    """
    https://ieeexplore.ieee.org/document/996017
    """
    # Sorting by multiple objectives:
    frontieer = fast_non_dominated_sort(solutions, domination_operator)

    # Rewarding diversity by boosting most different types:
    crowding_distance_assignment(solutions, objectives)

    solutions.sort(
        reverse=True,
        key=functools.cmp_to_key(
            lambda p, q: 1
            if p.rank < q.rank or (p.rank == q.rank and p.distance > q.distance)
            else -1
        ),
    )

    # Remove temporary values used by sorting algorithm:
    for solution in solutions:
        del solution.rank
        del solution.distance
        del solution.dominates
        del solution.dominated_by
    return solutions
