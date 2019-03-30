from src.buildingblocks.base import Base


def is_before(a, b) -> (Base, Base):
    """
    :param a: Node
    :param b: Node
    :return: True if a is before b in directed graph genotype
    """

    def match(node, target):
        if target in node.prev:
            return True
        return any(
            match(prev, target)
            for prev in node.prev
        )
    return match(a, b)



def connect(a, b):
    a, b = (b, a) if a .is_before(a, b) else (a, b)

    if not a in b.prev:
        b.prev += [a]
    if not b in a.next:
        a.next += [b]
