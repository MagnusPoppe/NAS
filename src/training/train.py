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


def stalling(history, steps=10, max_steps=100):
    if len(history["acc"]) < steps + 1:
        return False
    if len(history["acc"]) >= max_steps:
        return True

    acc = history["acc"][-1]
    vacc = history["val_acc"][-1]
    avg_acc = sum(history["acc"][-steps:-1]) / len(history["acc"][-steps:-1])
    avg_vacc = sum(history["val_acc"][-steps:-1]) / len(history["val_acc"][-steps:-1])

    # Average of previous steps within x % of average
    stale = avg_vacc * 0.985 <= vacc <= avg_vacc * 1.015 \
            and avg_acc * 0.985 <= acc <= avg_acc * 1.015 \
            and vacc < acc

    early_acc = history["acc"][int(steps * 0.3)]
    early_vacc = history["val_acc"][int(steps * 0.3)]
    no_improvement = early_vacc * 0.90 <= vacc <= early_vacc * 1.10 \
                     and early_acc * 0.90 <= acc <= early_acc * 1.10

    if not no_improvement:
        print(
            "\t- Found plateauing step for" if stale else "\t- Continuing training. State:",
            f"\n\t\tAccuracy:            {avg_acc * 0.98:.4} <= {acc:.4} <= {avg_acc * 1.02:.4}"
            f"\n\t\tValidation accuracy: {avg_vacc * 0.98:.4} <= {vacc:.4} <= {avg_vacc * 1.02:.4}"
        )
    else:
        print(
            f"\t- Stopping training due to No improvement {'and stale' if stale else ''}",
            f"(early={early_vacc}, latest={vacc})"
        )

    return stale or no_improvement


def train_until_stale(
        model,
        device,
        epochs,
        data,
        labels,
        val_data,
        val_labels,
        batch_size=64,
        compiled=False
):
    history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}

    with tf.device(device.device):
        # DEFINING FUNCTIONS FOR COMPILATION
        if not compiled:
            optimizer = keras.optimizers.Adam(lr=0.001)
            loss = keras.losses.categorical_crossentropy
            model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        # RUNNING TRAINING:
        while not stalling(history):
            metric = model.fit(
                x=data,
                y=labels,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True,
                validation_data=(val_data, val_labels),
            )
            # Recording training data:
            history["acc"] += metric.history["acc"]
            history["val_acc"] += metric.history["val_acc"]
            history["loss"] += metric.history["loss"]
            history["val_loss"] += metric.history["val_loss"]
    return metric.history
