import time

from src.buildingblocks.ops.operation import Operation

counter = 0

class Conv2D(Operation):

    def __init__(self, ID, kernel, filters, strides=(1, 1), activation="relu", bias=True):
        super().__init__(ID)
        self.kernel = kernel
        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = "same"
        self.bias = bias

    def to_keras(self):
        from tensorflow import keras
        return keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.bias,
            name=self.ID
        )

    def transfer_values(self, conv2D):
        conv2D.kernel = self.kernel
        conv2D.filters = self.filters
        conv2D.activation = self.activation
        conv2D.bias = self.bias
        conv2D.ID = self.ID
        return conv2D

class Conv3x3(Conv2D):

    def __init__(self, ID=None, kernel=(3, 3), filters=50):
        if not ID:
            global counter
            ID = "Conv3x3_{}".format(counter)
            counter += 1
        super().__init__(ID, kernel, filters)

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(Conv3x3())


class Conv5x5(Conv2D):

    def __init__(self, ID=None, kernel=(5, 5), filters=50):
        if not ID:
            global counter
            ID = "Conv5x5_{}".format(counter)
            counter += 1
        super().__init__(ID, kernel, filters)

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(Conv5x5())

