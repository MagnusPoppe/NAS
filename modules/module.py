
class Module():
    """
    Module is a collection of one or more modules and operations
    """

    ID = ""  # Should be set random by the app.

    def __init__(self):
        self.children = []

    def compile(self):
        """
        Converts the module's operations into actual keras operations
        in sequence.
        :return: tf.keras.model.Sequential
        """
        pass


class Operation():
    """
    An operation is a neural network operation, convertable to
    an actual object in a neural network.
    """

    ID = ""
    compatibility = []

    def __init__(self, ID, compatibility):
        self.next = []  # len() > 1: represents a split in the graph
        self.prev = []  # len() > 1: represents a merge in the graph

    def to_keras(self):
        """
        Converts the operation into a keras operation compatible with
        tf.keras.model.Sequential.
        """
        pass