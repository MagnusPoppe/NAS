from src.buildingblocks.module import Module


class Result():
    def __init__(self, accuracy, val_accuracy, loss, val_loss, learning_rate, distance, report, preferred_inputs):
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.loss = loss
        self.val_loss = val_loss
        self.learning_rate = learning_rate
        self.distance = distance
        self.report = report
        self.preferred_inputs = preferred_inputs
        self.model_path = ""

    def score(self):
        return self.accuracy[-1] * 0.2 \
               + self.val_accuracy[-1] * 0.4 \
               + self.report['weighted avg']['f1-score'] * 0.4


def apply_result(net, result, learning_rate):
    for dist, pattern in enumerate(net.patterns):
        pattern.results += [
            Result(
                accuracy=result['accuracy'],
                val_accuracy=result['validation accuracy'],
                loss=result['test accuracy'],
                val_loss=result['validation loss'],
                learning_rate=learning_rate,
                distance=dist / len(net.patterns),
                report=result['report'],
                preferred_inputs=net.patterns[dist - 1].find_last() if dist > 0 else None
            )
        ]


def inherit_results(patterns, nets):
    for pattern in patterns:
        successor_patterns = [
            p for net in nets
            for p in net.patterns
            if p.predecessor.ID == pattern.ID
        ]

        for successor in successor_patterns:
            pattern.results += successor.results
    return patterns
