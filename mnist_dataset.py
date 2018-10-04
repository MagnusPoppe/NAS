import os
import time


def _fix(data):
    import numpy as np
    return np.reshape(data, (len(data), 784))


def train(args) -> float:
    folder, epochs, batch_size, classes, device_name = args
    # Loading Model:
    import json
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(folder + "model.h5")

    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

    # VALIDATION DATA:
    x_val = _fix(x_train[50000:])
    y_val = y_train[50000:]
    x_val = x_val.astype('float32')
    x_val /= 255

    # TRAINING DATA:
    x_train = _fix(x_train[:50000])
    y_train = y_train[:50000]
    x_train = x_train.astype('float32')
    x_train /= 255


    # Converting to one-hot targets:
    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)


    # RUNNING TRAINING:
    with get_session(folder) as sess:
        sess.run(tf.global_variables_initializer())
        keras.backend.set_session(sess)

        # DEFINING FUNCTIONS AND COMPILING
        os.makedirs(os.path.join(folder, "sessions"))
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=0.01),
            metrics=['accuracy']
        )
        with tf.device(device_name):
            metrics = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(x_val, y_val)
            )
        print("    - Trained model...")
        saver = tf.train.Saver(sharded=True)
        saver.save(sess, os.path.join(folder, "sessions", "checkpoint.ckpt"))
    return metrics.history['val_acc'][-1]


def get_session(session_dir=None):
    import tensorflow as tf
    import os

    if session_dir and os.path.exists(session_dir + "sessions"):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph(os.path.join(session_dir, "sessions", "checkpoint.ckpt.meta"))
        saver.restore(
            sess=sess,
            save_path=os.path.join(session_dir, "sessions", "checkpoint.ckpt")
        )
    else:
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True),
        ))
    return sess


def evaluate(model, folder, classes=10) -> float:
    import tensorflow as tf
    from tensorflow import keras
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    # TEST DATA:
    x_test = _fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy
    with get_session(folder) as sess:
        sess.run(tf.global_variables_initializer())
        keras.backend.set_session(sess)
        model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
        metrics = model.evaluate(x_test, y_test, verbose=0)
    return metrics[1]  # Accuracy
