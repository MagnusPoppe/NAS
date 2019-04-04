import os
import pickle
import tensorflow as tf
from tensorflow import keras

from src.training import cifar10
from src.training.prepare_model import get_model
from src.training import train as training
from src.training.evaluate import evaluate


def module_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_new_session(device):
    with tf.device(device.device):
        keras.backend.set_session(
            tf.Session(
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(
                        allow_growth=device.allow_memory_growth,
                        per_process_gpu_memory_fraction=device.memory_per_process,
                    ),
                    allow_soft_placement=True
                )
            )
        )


def setup(individ, storage_directory, config, server_id, device_id):
    # Finding current compute device:
    device = config.servers[server_id].devices[device_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = device.device.split(":")[-1]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # Setting process name:
    try:
        import setproctitle
        setproctitle.setproctitle("NAS-TRAINER " + device.device)
    except ImportError:
        pass

    # Finding save directory and saving genotype:
    with open(os.path.join(storage_directory, "genotype.obj"), "wb") as f_ptr:
        pickle.dump(individ, f_ptr)

    set_new_session(device)
    return device


def finalize(individ, storage_directory, model, config):
    from src.pattern_nets import transer_knowlegde
    # Creating results:
    model_path = os.path.join(storage_directory, "model.h5")
    image_path = os.path.join(storage_directory, individ.ID + ".png")

    if config.type == "ea_nas":
        model_path = transer_knowlegde.try_save_model(model, model_path, individ.ID)
    else:
        transer_knowlegde.store_weights_in_patterns(individ, model, config)
        model_path = None
    try:
        keras.utils.plot_model(model, to_file=image_path)
    except Exception:
        pass
    return model_path, image_path


def run(args):
    # Unpacking arguments:
    individ_bytes, config_str, epochs, server_id, device_id, job_id = args
    individ = pickle.loads(individ_bytes)
    config = pickle.loads(config_str)
    storage_directory = individ.absolute_save_path(config)

    # Running setup
    device = setup(individ, storage_directory, config, server_id, device_id)

    # Getting or creating models:
    compiled, model = get_model(individ, config, device)

    # Running training:
    dataset = module_from_file(config.dataset_file_name, config.dataset_file_path)
    training_args = (
        model,
        device,
        epochs,
        *dataset.get_training_data(),
        *dataset.get_validation_data(),
        config.training.batch_size,
        compiled
    )
    if config.training.use_restart:
        training_history = training.train_until_stale(*training_args)
    else:
        training_history = training.train(*training_args)
    report = evaluate(model, *dataset.get_test_data(), device)

    # Finalizing and storing results:
    model_path, image_path = finalize(individ, storage_directory, model, config)
    return {
        "job": job_id,
        "image": image_path,
        "model": model_path,
        "epochs": epochs,
        "accuracy": training_history["acc"],
        "validation accuracy": training_history["val_acc"],
        "test accuracy": report['weighted avg']['precision'],
        "loss": training_history["loss"],
        "validation loss": training_history["val_loss"],
        "report": report
    }
