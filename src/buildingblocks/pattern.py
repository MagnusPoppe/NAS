from src.buildingblocks.module import Module


class Pattern(Module):

    def __init__(self, name=None, type: str = "1D", layers: int = 1):
        super().__init__(name)
        self.type = type
        self.layers = layers
        self.distance = 0
        self.preferred_distance = None
        self.results = []

    def __deepcopy__(self, memodict={}, clone=None):
        new = super().__deepcopy__(clone=Pattern())
        new.type = self.type
        new.layers = self.layers
        new.distance = self.distance
        new.preferred_distance = self.preferred_distance
        new.results = self.results
        return new
