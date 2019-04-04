import time

from src.buildingblocks.ops.operation import Operation

counter = 0


class Conv2D(Operation):

    def __init__(
            self,
            ID,
            kernel,
            filters,
            strides=(1, 1),
            activation="relu",
            bias=True,
            batch_norm=True,
            dropout=False,
            dropout_probability=0.3
    ):
        super().__init__(ID)
        self.kernel = kernel
        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = "same"
        self.bias = bias

        # Regularizer, Dropout and batch normalization are mutex.
        # If both are set, batch norm will be selected:
        if dropout and not batch_norm:
            self.dropout = dropout
            self.dropout_probability = dropout_probability
        elif batch_norm:
            self.dropout = False
            self.dropout_probability = 0.0
            self.batch_norm = batch_norm
        else:
            self.dropout = False
            self.dropout_probability = 0.0
            self.batch_norm = False

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

    def set_new_id(self):
        global counter
        name = self.ID.split("_")[0].strip()
        self.ID = f"{name}_{counter}"
        counter += 1

class Conv3x3(Conv2D):

    def __init__(self, ID=None, kernel=(3, 3), filters=50, dropout=True, dropout_probability=0.3):
        if not ID:
            global counter
            ID = "Conv3x3_{}".format(counter)
            counter += 1
        super().__init__(
            ID,
            kernel,
            filters,
            batch_norm=True,
            dropout=dropout,
            dropout_probability=dropout_probability
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(Conv3x3())


class Conv5x5(Conv2D):

    def __init__(self, ID=None, kernel=(5, 5), filters=50, dropout=True, dropout_probability=0.3):
        if not ID:
            global counter
            ID = "Conv5x5_{}".format(counter)
            counter += 1
        super().__init__(
            ID,
            kernel,
            filters,
            batch_norm=True,
            dropout=dropout,
            dropout_probability=dropout_probability
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(Conv5x5())
