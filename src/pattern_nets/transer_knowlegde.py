import time

import os

from src.output import no_stdout


def find_input_shape(config, pattern):
    if pattern.type == "1D" and len(config.input_format) < 2:
        input_dims = config.input_format
    elif pattern.type == "1D":
        _sum = 1
        for dim in config.input_format:
            _sum *= dim
        input_dims = [_sum]
    elif pattern.type == "2D" and len(config.input_format) > 2:
        input_dims = config.input_format
    else:
        raise ValueError("Wrong dimentions of input")
    return tuple(input_dims)


@no_stdout
def store_weights_in_patterns(individ, model, config):
    from src.frameworks.keras import module_to_model
    from src.frameworks.weight_transfer import transfer_model_weights_to_pattern

    for pattern in individ.patterns:
        input_dims = find_input_shape(config, pattern)
        pattern.detach()
        pattern_model = module_to_model(pattern, input_dims, config.classes_in_classifier)
        pattern.failed_transfers = transfer_model_weights_to_pattern(model, pattern_model)

        model_path = os.path.join(pattern.absolute_save_path(config), "model.h5")
        pattern.results[-1].model_path = try_save_model(pattern_model, model_path, pattern.ID)


def try_save_model(model, model_path, identity):
    """ Saving models will sometimes run out of workers.
        I cant control how others use the servers, so
        just retry up to 3 times and give up if not able
        to save.
    """
    from tensorflow import keras
    saved = False
    tries = 0
    while not saved:
        try:
            keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
            saved = True
        except OSError:
            tries += 1
            print("   - Failed to save model, retrying...")
            if tries == 3:
                print(f"   - Failed to save model for {identity}... Training data lost.")
                return None
            time.sleep(1)
    return model_path
