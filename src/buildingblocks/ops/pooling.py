import time

from src.buildingblocks.ops.operation import Operation

counter = 0
class Pooling(Operation):

    def __init__(self, ID, pool_size, strides=(1, 1)):
        ID = ID if ID else "MaxPooling2D_{}".format(time.time())
        super().__init__(ID)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = "same"

    def to_keras(self):
        raise NotImplementedError("Used the wrong pooling class... Use MaxPooling2x2 or AvgPooling2x2")

    def transfer_values(self, pooling):
        pooling.pool_size = self.pool_size
        pooling.strides = self.strides
        pooling.padding = self.padding
        pooling.ID = self.ID
        return pooling

class MaxPooling2x2(Pooling):

    def __init__(self):
        global counter
        ID = "MaxPooling2x2_{}".format(counter)
        counter += 1
        pool_size = (2, 2)
        super().__init__(ID, pool_size)

    def __deepcopy__(self, memodict={}):
        pooling = MaxPooling2x2()
        return self.transfer_values(pooling)

    def to_keras(self):
        from tensorflow import keras
        return keras.layers.MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            name=self.ID
        )


class AvgPooling2x2(Pooling):

    def __init__(self):
        global counter
        ID = "AvgPooling2x2_{}".format(counter)
        counter += 1
        pool_size = (2, 2)
        super().__init__(ID, pool_size)

    def __deepcopy__(self, memodict={}):
        pooling = AvgPooling2x2()
        return self.transfer_values(pooling)

    def to_keras(self):
        from tensorflow import keras
        return keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            name=self.ID
        )

