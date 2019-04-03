from src.buildingblocks.module import Module


class Pattern(Module):

    def __init__(self, name=None, type: str = "1D", layers: int = 1):
        super().__init__(name)
        self.type = type
        self.layers = layers
        self.placement = 0
        self.preferred_placement = None
        self.results = []

    def __deepcopy__(self, memodict={}, clone=None):
        new = super().__deepcopy__(clone=Pattern())
        new.type = self.type
        new.layers = self.layers
        new.placement = self.placement
        new.preferred_placement = self.preferred_placement
        # new.results = self.results
        return new

    def detach(self):
        for child in self.children:
            removals = [n for n in child.next if n not in self.children]
            for rm in removals:
                rm.prev.remove(child)
                child.next.remove(rm)
            removals = [n for n in child.prev if n not in self.children]
            for rm in removals:
                rm.next.remove(child)
                child.prev.remove(rm)
