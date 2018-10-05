from operator import attrgetter

from tensorflow import keras

from modules.base import Base
from modules.module import Module
from frameworks.common import rank_children

def assemble(module:Module, in_shape:tuple=(784,), classes:int=10, is_root:bool=True):

    # 1. Rank and sort all child operations using breadth-first:
    rank_children(module)
    operations = sorted(module.children, key=attrgetter('rank'))

    # 2. Connect keras operations together:
    input = keras.layers.Input(shape=in_shape)
    for node in operations:
        node.keras_tensor = connect_operation_to_previous(node, node.prev, input)

    # 3. Create a trainable keras.models.Model for module:
    module.keras_operation = keras.models.Model(inputs=[input], outputs=[
        keras.layers.Dense(units=classes, activation="softmax")(operations[-1].keras_tensor) if is_root
        else operations[-1].keras_tensor
    ])
    return module.keras_operation


def connect_operation_to_previous(node: Base, previous: list, input_layer: keras.layers.Input):
    """ Connects a (module / operation) to its previous nodes. This performs
        a recursive call to decode if the node is type Module. Then applies
        the keras operation.
        :returns: The output tensor of this operation
    """
    if len(previous) == 0:
        # First node uses the input layer as previous
        previous_output = input_layer
    elif len(previous) == 1:
        # Regular input
        previous_output = previous[0].keras_tensor
    else:
        # Concatenation of all inputs
        previous_output = keras.layers.concatenate([_prev.keras_tensor for _prev in previous])
    if isinstance(node, Module):
        return assemble(node, in_shape=previous_output.shape, is_root=False)(previous_output)

    # else: Operation
    return node.to_keras()(previous_output)
