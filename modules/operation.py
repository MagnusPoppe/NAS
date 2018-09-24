from modules.base import Base


class Operation(Base):
    """
    An operation is a neural network operation, convertable to
    an actual operation in a neural network.

    **NOTE**:
    Instances of this class should always have a parent module.
    """

    def __init__(self, ID, compatible):
        super().__init__()
        self.ID = ID
        self.keras_operation = None
        self.compatible = compatible

    def __str__(self):
        return self.ID

    def to_keras(self):
        """
        Converts the operation into a keras operation compatible with
        tf.keras.model.Sequential.
        """
        pass