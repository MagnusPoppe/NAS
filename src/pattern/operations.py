
def swap_if_before(a, b) -> tuple:
    """
    :param a: Node
    :param b: Node
    :return: True if a is before b in directed graph
    """
    def match(node, target):
        if target in node.prev:
            return True
        return any(
            match(prev, target)
            for prev in node.prev
        )
    return b, a if match(a, b) else a, b


def connect(a, b):
    a, b = swap_if_before(a, b)
    a.next += [b]
    b.prev += [a]
