
def classification_objectives(config) -> [callable]:
    def get_precision(i):
        def _get_report(p):
            return p.results[-1].report[str(i)]['f1-score']
        return _get_report
    return [get_precision(cls) for cls in range(config.classes_in_classifier)]


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
