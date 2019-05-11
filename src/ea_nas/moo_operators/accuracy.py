
def objectives(*args, **kwargs) -> [callable]:
    return [
        lambda p: abs(int(1000 * p.test_acc()) - int(1000 * p.acc())),
        lambda p: int(1000 * p.test_acc()),
        lambda p: p.number_of_operations(),
    ]

def domination_operator(_objectives:[callable]) -> callable:
    operators = _objectives

    def inner(p, q) -> bool:
        """
        This represents the crowded comparison operator p dominates q
        if BOTH conditions are met:
            1. p is no worse than q in all objectives
            2. p is strictly better than q on at least one objective
        """
        comparison = [
            operators[0](q) - operators[0](p),  # Less is better
            operators[1](p) - operators[1](q),  # More is better
            operators[2](q) - operators[2](p),  # Less is better
        ]

        # "Strictly better" and "no worse"
        return any(c > 0 for c in comparison) and all(c >= 0 for c in comparison)
    return inner
