from src.buildingblocks.base import Base


class Operation(Base):
    """
    An operation is a neural network operation, convertable to
    an actual operation in a neural network.

    **NOTE**:
    Instances of this class should always have a parent module.
    """

    def __init__(self, ID):
        super().__init__()
        self.ID = ID

    def __str__(self):
        return self.ID

    def __copy__(self):
        raise NotImplementedError("Copy function not yet implemented...")

    def to_keras(self):
        """
        Converts the operation into a keras operation compatible with
        tf.keras.model.Sequential.
        """
        pass
