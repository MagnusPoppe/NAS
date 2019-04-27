def adjust_learning_rate(data: [(float, float)]):
    """
    Adjusts learning rate according to the weighted average
    slope found towards the best scored learning rate.
    :param data: List of tuples containing (accuracy, learning_rate)
    :return: Adjusted learning rate
    """

    def slope(x1, y1, x2, y2) -> float:
        return (y2 - y1) / (x2 - x1)

    data.sort(key=lambda x: x[1])
    b_acc, b_lr = data.pop(-1)

    # Finding weights to use for learning rates
    sum_acc = sum([acc for acc, _ in data])
    weights = [acc / sum_acc for acc, lr in data]

    # Calculating weighted average slope
    slopes = [slope(b_acc, b_lr, acc, lr) * weight for (acc, lr), weight in zip(data, weights)]
    avg_adjustment = sum(slopes) / len(slopes)

    # Calculating the new learning rate
    return b_lr + (avg_adjustment * b_lr)


def set_learning_rate(net):
    if all(p.used_result for p in net.patterns):
        args = [
            (p.used_result.score(), p.used_result.learning_rate)
            for p in net.patterns
        ]
        if args:
            net.learning_rate = adjust_learning_rate(args)
