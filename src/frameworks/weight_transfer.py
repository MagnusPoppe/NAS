from src.buildingblocks.module import Module


def transfer_model_weights(model, old_model) -> Module:
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError.
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """

    for from_layer in old_model.layers:
        for layer in model.layers:
            if layer.name == from_layer.name:
                try:
                    layer.set_weights(from_layer.get_weights())
                except ValueError as e:
                    print("Weight incompatability: " + str(e))


def transfer_predecessor_weights(module, predecessor) -> Module:
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