from src.buildingblocks.module import Module
from src.pattern_nets.evaluation import Result


class Pattern(Module):

    def __init__(self, name=None, type: str = "1D", layers: int = 1):
        super().__init__(name)
        self.type = type
        self.layers = layers
        self.placement = 0
        self.preferred_placement = None
        self.used_result = None
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

    def optimal_result(self):
        def is_better(this: Result, best: Result):
            return (
                this.val_accuracy[-1] >= best.val_accuracy[-1] and
                this.accuracy[-1] >= best.accuracy[-1] and
                this.report['weighted avg']['f1-score'] >= best.report['weighted avg']['f1-score']
            )

        if len(self.results) > 0:
            best_res = self.results[0]
            for res in self.results[1:]:
                if is_better(res, best_res):
                    best_res = res
            return best_res
        return None

    def misconfigured(self):
        try:
            _ = self.type
            _ = self.layers
        except AttributeError:
            return True
        try:
            _ = self.placement
        except AttributeError:
            self.placement = 0
        try: _ = self.preferred_placement
        except AttributeError:
            self.preferred_placement = None
        try: _ = self.used_result
        except AttributeError:
            self.used_result = None
        try: _ = self.results
        except AttributeError:
            self.results = []

        return False
