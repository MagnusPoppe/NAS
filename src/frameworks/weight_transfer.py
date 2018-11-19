from src.buildingblocks.module import Module


def transfer_model_weights(model, old_model):
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError.
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """
    from tensorflow import keras

    for from_layer in old_model.layers:
        for layer in model.layers:
            if layer.name == from_layer.name:
                if isinstance(layer, keras.models.Model):
                    # Transfer learning on sub-modules:
                    transfer_model_weights(layer, from_layer)
                else:
                    try:
                        layer.set_weights(from_layer.get_weights())
                    except ValueError as e:
                        print("Weight incompatability: " + str(e))


def transfer_predecessor_weights(module, predecessor):
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError.
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """
    # Searching for matching operators in both module and modules predecessor
    # to find out where to transfer weights from.
    for predecessor_child in predecessor.children:
        for child in module.children:
            if child.ID == predecessor_child.ID:
                try:
                    # Copying weights:
                    child.keras_operation.set_weights(
                        predecessor_child.keras_operation.get_weights()
                    )
                except ValueError:
                    # Not all layers can be copied over due to mutations in the
                    # network architecture. Ignore the weights that this applies to.
                    break
    return module

def transfer_submodule_weights(module: Module):
    """
    Transfer weights for sub-modules where the sub-module is not included
    in the predecessor.

    The standard transfer_model_weights() algorithm transfers all weights from
    predecessor model to the current model. A sub-module may have been inserted
    into this current module. We need to transfer weights for this sub-module as
    well
    :param module: to transfer from
    """

    from tensorflow import keras
    # Excluding modules that have trained as part of predecessor network:
    exclusion = [op for op in module.predecessor if isinstance(op, Module)]

    # Finding sub-modules to transfer weights to:
    sub_modules = [op for op in module.children if isinstance(op, Module) and op not in exclusion]

    # Transferring weights:
    for sub_module in sub_modules:
        pred_model = keras.models.load_model(sub_module.predecessor.saved_model)
        model = sub_module.keras_operation
        transfer_model_weights(model, pred_model)