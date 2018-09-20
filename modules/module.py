from modules.base import Base
from modules.operation import Operation


class Module(Base):
    """
    Module is a collection of one or more modules and operations
    """

    ID = ""  # Should be set random by the app.

    def __init__(self):
        super().__init__()
        self.children = []

    def __iadd__(self, other):
        if isinstance(other, Operation) or isinstance(other, Module):
            if len(self.children) < 1:
                self.children += [other]
            else:
                previous = self.children[-1]
                previous.next += [other]
                other.prev += [previous]
                self.children += [other]
        return self

    def __str__(self):
        return "Module [{}]".format(", ".join([str(c) for c in self.children]))

    def visualize(self):
        pass

    def compile(self):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :return: tf.keras.model.Sequential
        """
        pass