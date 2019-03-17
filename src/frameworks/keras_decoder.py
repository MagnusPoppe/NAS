from operator import attrgetter

from tensorflow import keras

from src.buildingblocks.base import Base
from src.buildingblocks.ops.convolution import Conv2D
from src.buildingblocks.ops.dense import Dense # , Dropout
from src.buildingblocks.module import Module
from src.evolutionary_operations.mutation_for_operators import is1D, is2D
from src.frameworks.common import rank_children

def assemble(module:Module, in_shape:tuple=(784,), classes:int=10, is_root:bool=True, indent=""):
    # 1. Rank and sort all child operations using breadth-first:
    rank_children(module)
    operations = sorted(module.children, key=attrgetter('rank'))

    # 2. Connect keras operations together:
    if not isinstance(in_shape[0], int) and not is_root:
        in_shape = tuple(dim.value for dim in in_shape if dim.value)
    input = keras.layers.Input(shape=in_shape)

    for node in operations:
        node.keras_tensor = connect_operation_to_previous(node, node.prev, input, indent)

    # FLATTENING ANY CONV OUTPUT:
    previous_tensor = operations[-1].keras_tensor

    # DEFINING OUTPUT FOR THE MODEL:
    if is_root:
        if len(previous_tensor.shape) > 2:
            previous_tensor = keras.layers.Flatten()(previous_tensor)
        output = keras.layers.Dense(units=classes, activation="softmax")(previous_tensor)
    else:
        output = previous_tensor
    # 3. Create a trainable keras.models.Model for module:
    try:
        module.keras_operation = keras.models.Model(
            inputs=[input],
            outputs=[output],
            name="-".join(module.ID.split())
        )
    except ValueError as e:
        print(indent + "    - Crashed with input: {} and output: {}".format(input.name, output.shape))
        raise e
    return module.keras_operation


def connect_operation_to_previous(node: Base, previous: list, input_layer: keras.layers.Input, indent):
    """ Connects a (module / operation) to its previous nodes. This performs
        a recursive call to decode if the node is type Module. Then applies
        the keras operation.
        :returns: The output tensor of this operation
    """

    if len(previous) == 0:
        # First node uses the input layer as previous
        previous_output = input_layer
    else:
        tensors = get_tensors(previous, node)
        if len(previous) == 1:
            # Regular input
            previous_output = tensors[0]
        else:
            # Concatenation of all inputs
            previous_output = keras.layers.concatenate([tensor for tensor in tensors])

    if isinstance(node, Module):
        return assemble(node, in_shape=previous_output.shape, is_root=False, indent=indent+"    ")(previous_output)
    else:  # Operation
        node.keras_operation = node.to_keras()
        return node.keras_operation(previous_output)

def fix_padding(tensor, target_shape):
    """ Fixes a problem that occurs with when using """
    pass

def get_tensors(previous, current):
    def needs_flatten(prev):
        return is1D(current) and is2D(prev) \
            or (isinstance(prev, Module) and is2D(prev.find_last()[0]))

    tensors = []
    for prev in previous:
        if needs_flatten(prev):
            flatten = keras.layers.Flatten()(prev.keras_tensor)
            tensors += [flatten]
            # print("    - Flatten layer from {} to {}".format(prev.keras_tensor.shape, flatten))
        else:
            tensors += [prev.keras_tensor]
    return tensors
