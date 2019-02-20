def acc_objectives() -> [callable]:
    return [
        lambda p: abs(int(1000 * p.validation_fitness[-1]) - int(1000 * p.fitness[-1])),
        lambda p: abs(int(1000 * p.validation_loss[-1]) - int(1000 * p.loss[-1])),
        lambda p: int(1000 * p.validation_fitness[-1]),
        lambda p: p.number_of_operations(),
    ]


def classification_objectives(config) -> [callable]:
    def get_precision(i):
        return lambda p: p.report[p.epochs_trained][str(i)]['f1-score']
    return [get_precision(cls) for cls in range(config['classes'])]


def classification_domination_operator(objectives: [callable]) -> callable:
    operators = objectives

    def inner(p, q) -> bool:
        """
        This represents the crowded comparison operator p dominates q
        if BOTH conditions are met:
            1. p is no worse than q in all objectives
            2. p is strictly better than q on at least one objective
        """
        comparison = [op(p) - op(q) for op in operators]

        # "Strictly better" and "no worse"
        return any(c > 0 for c in comparison) and all(c >= 0 for c in comparison)

    return inner


def acc_domination_operator(objectives:[callable]) -> callable:
    operators = objectives

    def inner(p, q) -> bool:
        """
        This represents the crowded comparison operator p dominates q
        if BOTH conditions are met:
            1. p is no worse than q in all objectives
            2. p is strictly better than q on at least one objective
        """
        comparison = [
            operators[0](q) - operators[0](p),  # Less is better
            operators[1](q) - operators[1](p),  # Less is better
            operators[2](p) - operators[2](q),  # More is better
            operators[3](q) - operators[3](p),  # Less is better
        ]

        # "Strictly better" and "no worse"
        return any(c > 0 for c in comparison) and all(c >= 0 for c in comparison)
    return inner
