class Base():

    def __init__(self):
        self.nodeID = None
        self.next = []  # len() > 1: represents a split in the graph
        self.prev = []  # len() > 1: represents a merge in the graph

    def to_keras(self):
        raise NotImplementedError("Not yet implemented to_keras method for this class...")

    def disconnect(self):
        for p in self.prev:
            if self in p.next:
                p.next.remove(self)
        for n in self.next:
            if self in n.prev:
                n.prev.remove(self)

    def ensure_connected(self):
        for p in self.prev:
            if not self in p.next:
                p.next += [self]
        for n in self.next:
            if not self in n.prev:
                n.prev += [self]

    def inherit_connectivity_from(self, old):
        self.next = old.next
        self.prev = old.prev
        old.disconnect()
        self.ensure_connected()

    def is_after(self, b) -> bool:
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
        return match(self, b)

    def connect_to(self, b):
        self, b = (b, self) if self.is_after(b) else (self, b)

        if not self in b.prev:
            b.prev += [self]
        if not b in self.next:
            self.next += [b]
