from src.helpers import randomized_index


def tournament(population, size):
    index = list(zip(
        randomized_index(population, index_size=size),
        randomized_index(population, index_size=size)
    ))
    while any([x == y for x, y in index]):
        index = list(zip(
            randomized_index(population, index_size=size),
            randomized_index(population, index_size=size)
        ))

    selected = []
    for x, y in index:
        x_score = population[x].optimal_result().report["weighted avg"]["f1-score"]
        y_score = population[y].optimal_result().report["weighted avg"]["f1-score"]
        selected += [population[x] if x_score > y_score else population[y]]
    return selected


def divide(selected):
    import random
    # Finding 1D and 2D
    li1D, li2D = [], []
    for i in range(len(selected)):
        if selected[i].type == "2D":
            li2D += [selected[i]]
        else:
            li1D += [selected[i]]

    # Distribution:
    draw = random.uniform(0, 1)
    cross_overs = int(len(selected) * (1 - draw))
    mutations = int(len(selected) * draw)
    if cross_overs % 2 == 1:  # Cannot be odd number...
        cross_overs -= 1
        mutations += 1

    cross_over_candidates = []
    for i in range(0, cross_overs, 2):
        selector = li1D
        if len(li1D) <= 1 and 1 >= len(li2D):
            break
        elif len(li2D) <= 1:
            selector = li1D
        elif len(li1D) <= 1:
            selector = li2D
        elif random.uniform(0, 1) > 0.5:
            selector = li2D

        cross_over_candidates += [(selector.pop(0), selector.pop(0))]

    mutation_candidates = li2D + li1D
    return mutation_candidates, cross_over_candidates
