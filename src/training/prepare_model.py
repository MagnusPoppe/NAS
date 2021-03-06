import os
import tensorflow as tf
from tensorflow import keras
from src.frameworks.keras import module_to_model as assemble
from src.frameworks.weight_transfer import transfer_model_weights
from src.output import no_stdout


@no_stdout
def get_model(individ, config, device):
    if config.type == "PatternNets":
        return prepare_pattern_model(individ, config, device)
    else:
        return prepare_ea_nas_model(individ, config, device)


def prepare_ea_nas_model(individ, config, device):
    compiled = False

    # Checking if already trained:
    if individ.saved_model and os.path.isfile(individ.saved_model):
        with tf.device(device):
            model = keras.models.load_model(individ.saved_model)
        compiled = True
    # Is not trained. Apply transfer learning if possible:
    else:
        with tf.device(device):
            model = assemble(individ, config.input_format, config.classes_in_classifier)

            # Can transfer learn?
            if config.use_transfer_learning:
                try:
                    if individ.predecessor:
                        file = individ.predecessor.saved_model
                        if file and os.path.isfile(file):
                            predecessor_model = keras.models.load_model(individ.predecessor.saved_model)
                            transfer_model_weights(model, predecessor_model)
                except KeyError:
                    print("Could not use transfer learning due to bad file...")
    return compiled, model


def prepare_pattern_model(individ, config, device):
    from src.frameworks.keras import module_to_model as assemble
    model = assemble(individ, config.input_format, config.classes_in_classifier)

    with tf.device(device):
        for pattern in individ.patterns:
            pattern.model_file_exists(config)
            if pattern.used_result and pattern.used_result.model_path:
                pattern_model = keras.models.load_model(pattern.used_result.model_path)
                transfer_model_weights(model, pattern_model)

    return False, model
