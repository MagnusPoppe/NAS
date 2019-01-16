
def objectives():
    return [
        lambda p: abs(
            int(1000 * p.validation_fitness[-1]) - int(1000 * p.fitness[-1])),
        lambda p: abs(
            int(1000 * p.validation_loss[-1]) - int(1000 * p.loss[-1])),
        lambda p: int(1000 * p.validation_fitness[-1]),
        lambda p: p.number_of_operations()
    ]


def domination_operator(p, q):
    operators = objectives()
    comparison = [
        operators[0](q) - operators[0](p),  # Less is better
        operators[1](q) - operators[1](p),  # Less is better
        operators[2](p) - operators[2](q),  # More is better
        operators[3](q) - operators[3](p),  # Less is better
    ]
    return any(c > 0 for c in comparison) and all(c >= 0 for c in comparison)
