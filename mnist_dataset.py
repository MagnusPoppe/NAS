import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

available_devices = ['/device:GPU:1']
def generate_device_list() -> list:
    dev = 0
    while True:
        yield available_devices[dev % len(available_devices)]
        dev += 1


def mnist_configure(classes):  # -> (function, function):
    def fix(data):
        return np.reshape(data, (len(data), 784))

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # VALIDATION DATA:
    x_val = fix(x_train[50000:])
    y_val = y_train[50000:]
    x_val = x_val.astype('float32')
    x_val /= 255

    # TRAINING DATA:
    x_train = fix(x_train[:50000])
    y_train = y_train[:50000]
    x_train = x_train.astype('float32')
    x_train /= 255

    # TEST DATA:
    x_test = fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    # Converting to one-hot targets:
    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)

    # DEFINING FUNCTIONS AND COMPILING
    eval_loss = keras.losses.categorical_crossentropy
    eval_optimizer = keras.optimizers.Adam(lr=0.01)

    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True)
    ))
    keras.backend.set_session(session)
    devices = generate_device_list()


    # Setting up dataset:
    dataset = tf.contrib.data.Dataset.from_tensor_slices(x_train)

    def train(population: list, epochs=5, batch_size=64):
        started = time.time()
        # Preparing models for training:
        placeholders = []
        trainers = []
        labels = tf.placeholder(tf.float32, shape=(None, 10))
        for i, individ in enumerate(population):
            if not individ.device: individ.device = next(devices)
            with tf.device(individ.device):
                model = individ.keras_operation  # type: keras.models.Model
                loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(labels, model.output))
                optimizer = tf.train.AdamOptimizer(0.01)

                trainers += [optimizer.minimize(loss)]
                placeholders += [(model.input, labels)]

        # RUNNING TRAINING:
        keras.backend.learning_phase = True
        with session as sess:
            sess.run(tf.global_variables_initializer())

            prep_time = 0
            train_time = 0

            print("--> Training for {} epochs on {} models ".format(epochs, len(population)), end="", flush=True)
            for epoch in range(epochs):

                # DATASET PREPARATION:
                for i in range(0, len(x_train), batch_size):
                    start = time.time()
                    # Batch setup:
                    batch_end = batch_size * i + batch_size
                    batch_train = x_train[i:(batch_end if batch_end < len(x_train) else len(x_train))]
                    batch_labels = y_train[i:(batch_end if batch_end < len(y_train) else len(y_train))]
                    feeder = {}
                    for labels, cost in placeholders:
                        feeder[labels] = batch_train
                        feeder[cost] = batch_labels

                    # RUNNING TRAINING:
                    training_start = time.time()
                    prep_time += (training_start - start)
                    sess.run(trainers, feed_dict=feeder)
                    train_time += (time.time() - training_start)
                print("--> Epoch 1 complete! Time used on preparation: {}, time used on training: {}".format(prep_time, train_time))

        print("(elapsed time: {} sec)".format(int(time.time() - started)))
        print("--> Time used on preparation: {}, time used on training: {}".format(prep_time, train_time))
        evaluate(population)

    def evaluate(population: list):
        print("--> Evaluating {} models".format(len(population)))

        for individ in population:
            model = individ.keras_operation
            model.compile(loss=eval_loss, optimizer=eval_optimizer, metrics=['accuracy'])
            metrics = model.evaluate(x_test, y_test, verbose=0)
            individ.fitness = metrics[1]  # Accuracy

    return train, evaluate
