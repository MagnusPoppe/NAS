import tensorflow as tf
from tensorflow import keras


def train(
        model,
        device,
        epochs,
        data,
        labels,
        val_data,
        val_labels,
        batch_size=64,
        compiled=False,
):
    with tf.device(device.device):
        # DEFINING FUNCTIONS FOR COMPILATION
        if not compiled:
            optimizer = keras.optimizers.Adam(lr=0.001)
            loss = keras.losses.categorical_crossentropy
            model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        # RUNNING TRAINING:
        metric = model.fit(
            x=data,
            y=labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
            validation_data=(val_data, val_labels),
        )
    return metric.history
