
class Result():
    def __init__(self, distance, report, preferred_inputs):
        self.distance = distance
        self.report = report
        self.preferred_inputs = preferred_inputs


def apply_results(patterns, nets):
    for pattern in patterns:
        included_nets = [
            net for net in nets
            for p in net.patterns
            if p.predecessor.ID == pattern.ID
        ]

        for net in included_nets:
            i = [i for i, p in enumerate(net.patterns) if p.predecessor.ID == pattern.ID][0]

            pattern.results += [
                Result(
                    net.patterns[i].placement,
                    net.report[list(net.report.keys())[-1]],
                    net.patterns[i - 1].find_last() if i > 0 else None
                )
            ]

    return patterns

