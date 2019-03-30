
class Result():
    def __init__(self, distance, report):
        self.distance = distance
        self.report = report


def apply_results(patterns, nets):
    for net in nets:
        for i in range(len(net.patterns)):
            clone = net.patterns[i]
            original = [p for p in patterns if p.ID == clone.predecessor.ID][0]
            original.results += [Result(clone.distance, net.report[list(net.report.keys())[-1]])]
            if i > 0:
                original.preferred_inputs = net.patterns[i-1].find_last()
    return patterns
