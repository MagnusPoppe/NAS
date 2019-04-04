from src.buildingblocks.module import Module


class Result():
    def __init__(self, accuracy, val_accuracy, loss, val_loss, distance, report, preferred_inputs):
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.loss = loss
        self.val_loss = val_loss
        self.distance = distance
        self.report = report
        self.preferred_inputs = preferred_inputs

    def score(self):
        return self.accuracy[-1] * 0.2 \
            + self.val_accuracy[-1] * 0.4 \
            + self.report['weighted avg']['f1-score'] * 0.4


def apply_results(patterns, nets):
    for pattern in patterns:
        included_nets = [
            net for net in nets
            for p in net.patterns
            if p.predecessor.ID == pattern.ID
        ]

        for net in included_nets:
            dist = [i for i, p in enumerate(net.patterns) if p.predecessor.ID == pattern.ID][0]
            pattern.results += [
                Result(
                    net.fitness,
                    net.validation_fitness,
                    net.loss,
                    net.validation_loss,
                    dist / len(net.patterns),
                    net.report[list(net.report.keys())[-1]],
                    net.patterns[dist - 1].find_last() if dist > 0 else None
                )
            ]

    return patterns

