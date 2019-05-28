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
        return self.acc() * 0.2 \
               + self.val_acc() * 0.4 \
               + self.test_acc() * 0.4

    def acc(self):
        try:
            return self.accuracy[-1]
        except IndexError:
            return 0.0

    def val_acc(self):
        try:
            return self.val_accuracy[-1]
        except IndexError:
            return 0.0

    def test_acc(self):
        try:
            return self.report['weighted avg']["precision"]
        except (KeyError, IndexError):
            return 0.0


def apply_result(net, result, learning_rate):
    for dist, pattern in enumerate(net.patterns):
        preferred_inputs = None
        try:
            preferred_inputs = net.patterns[dist - 1].find_last() if dist > 0 else None
        except KeyError:
            print("Found KeyError")

        try:
            pattern.results += [
                Result(
                    accuracy=result['accuracy'],
                    val_accuracy=result['validation accuracy'],
                    loss=result['test accuracy'],
                    val_loss=result['validation loss'],
                    learning_rate=learning_rate,
                    distance=dist / len(net.patterns),
                    report=result['report'],
                    preferred_inputs=preferred_inputs
                )
            ]
        except KeyError:
            print(f"Results was lost for Pattern {pattern.ID}")


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
