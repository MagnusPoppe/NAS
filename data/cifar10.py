import numpy as np
from tensorflow import keras

# Preparing data:
TRAINING_CASES = 45000
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_val = x_train[TRAINING_CASES:] / 255
y_val = keras.utils.to_categorical(y_train[TRAINING_CASES:], num_classes=10)
x_train = x_train[:TRAINING_CASES] / 255
y_train = keras.utils.to_categorical(y_train[:TRAINING_CASES], num_classes=10)
x_test = x_test / 255


def get_training_data(augment=False) -> (np.ndarray, np.ndarray):
    global x_train, y_train
    if augment:
        aug = np.flip(x_train, 2)
        x_augmented = np.concatenate((x_train, aug), axis=0)
        y_augmented = np.concatenate((y_train, y_train), axis=0)
        return x_augmented, y_augmented
    return x_train, y_train


def get_validation_data() -> (np.ndarray, np.ndarray):
    return x_val, y_val


def get_test_data() -> (np.ndarray, np.ndarray):
    return x_test, y_test


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     print("Original length: ", len(x_train))
#     x_aug, y_aug = get_training_data(True)
#     print("x_train length: ", len(x_aug))
#     print("y_train length: ", len(y_aug))
#     plt.imshow(x_aug[29921])
#     print("train answer=")
#     print(y_aug[29921])
#     print("aug answer=")
#     print(y_aug[74921])
#     plt.show()