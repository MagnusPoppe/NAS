from frameworks.keras_decoder import assemble
from modules.module import Module


def transfer_predecessor_weights(module: Module, in_shape: tuple, classes: int, predecessor=None) -> Module:
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError.
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """
    predecessor = predecessor if predecessor else module.predecessor
    module.keras_tensor = assemble(module, in_shape, classes)

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