class Base():

    def __init__(self):
        self.nodeID = None
        self.next = []  # len() > 1: represents a split in the graph
        self.prev = []  # len() > 1: represents a merge in the graph

    def to_keras(self):
        raise NotImplementedError("Not yet implemented to_keras method for this class...")