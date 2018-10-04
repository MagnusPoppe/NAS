import os
import time

def load_session(session_dir=None):
    import tensorflow as tf
    from tensorflow import keras
    import os

    if os.path.exists(session_dir):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(session_dir, "checkpoint.ckpt.meta"))
        saver.restore(
            sess=sess,
            save_path=os.path.join(session_dir, "checkpoint.ckpt")
        )
    else:
        os.makedirs(session_dir, exist_ok=True)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True),
        ))

    # graph = tf.Graph().as_default()
    sess.run(tf.global_variables_initializer())
    keras.backend.set_session(sess)
    return sess


def save_session(session_dir, session):
    import tensorflow as tf
    os.makedirs(session_dir, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(session, os.path.join(session_dir, "checkpoint.ckpt"))


def _fix(data):
    import numpy as np
    return np.reshape(data, (len(data), 784))


def train(args) -> float:
    folder, epochs, batch_size, classes, device_name = args
    # Loading Model:
    import json
    import tensorflow as tf
    from tensorflow import keras

    session_folder = os.path.join(folder, "sessions")
    with load_session(session_folder) as sess:
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



        # DEFINING FUNCTIONS AND COMPILING
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=0.01),
            metrics=['accuracy']
        )
        with tf.device(device_name):
            # RUNNING TRAINING:
            metrics = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(x_val, y_val)
            )
        print("    - Trained model...")
        save_session(session_folder, sess)
        tf.keras.backend.clear_session()
    return metrics.history['val_acc'][-1]

def evaluate(model, folder, classes=10) -> float:
    import tensorflow as tf
    from tensorflow import keras
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    session_folder = os.path.join(folder, "sessions")
    # TEST DATA:
    x_test = _fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy
    with load_session(session_folder) as sess:
        tf.keras.backend.clear_session()
        model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
        metrics = model.evaluate(x_test, y_test, verbose=0)
    return metrics[1]  # Accuracy
