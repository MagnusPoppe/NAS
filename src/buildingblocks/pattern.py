from src.buildingblocks.module import Module


class Pattern(Module):

    def __init__(self, name=None, type: str = "1D", layers: int = 1):
        super().__init__(name)
        self.type = type
        self.layers = layers

    def __deepcopy__(self, memodict={}):
        new = super().__deepcopy__()
        new.type = self.type
        new.layers = self.layers
        return new
