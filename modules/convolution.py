from modules.module import Operation
from tensorflow import keras


class Conv2D(Operation):

    def __init__(self, ID, kernel, filters, strides=(1, 1), activation="ReLU", bias=True):
        super().__init__(ID, [type(self)])
        self.kernel = kernel
        self.filters = filters
        self.strides = strides
        self.activation = activation,
        self.bias = bias

    def to_keras(self):
        return keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel,
            strides=self.strides,
            activation=self.activation,
            use_bias=self.bias
        )


class Conv3x3(Conv2D):

    def __init__(self, ID="Conv3x3", kernel=(3, 3), filters=50):
        super().__init__(ID, kernel, filters)


class Conv5x5(Conv2D):

    def __init__(self, ID="Conv5x5", kernel=(5, 5), filters=50):
        super().__init__(ID, kernel, filters)
