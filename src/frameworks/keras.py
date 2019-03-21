from src.buildingblocks.module import Module
from src.buildingblocks.ops.convolution import Conv2D
from src.buildingblocks.ops.dense import Dense
from tensorflow import keras

from src.buildingblocks.ops.pooling import AvgPooling2x2, MaxPooling2x2
from src.ea_nas.evolutionary_operations.mutation_for_operators import is2D, is1D


def ensure_correct_tensor_by_shape(tensors):
    def flatten(_tensors):
        flattened = []
        for tensor in _tensors:
            if len(tensor.shape) > 2:
                flattened += [keras.layers.Flatten()(tensor)]
            else:
                flattened += [tensor]
        return flattened

    tensors = flatten(tensors)
    return tensors[0] if len(tensors) == 1 else keras.layers.Concatenate()(tensors)


def ensure_correct_tensor_by_op(op, inputs):
    # Fixing previous tensors:
    # Requirements:
    # 1. Must be single tensor. Multiple tensors must be concatinated.
    # 2. Must be of correct input shape.
    # 3. If no previous layer, use input layer.
    def correct_input_shapes(op):
        tensors = []
        if is1D(op) and any([is2D(p) for p in op.prev]):
            for prev_op in op.prev:
                if is2D(prev_op):
                    tensors += [keras.layers.Flatten()(prev_op.tensor)]
                else:
                    tensors += [prev_op.tensor]
            return tensors
        return [p.tensor for p in op.prev]

    tensors = correct_input_shapes(op)
    if len(tensors) > 1:
        prev = keras.layers.Concatenate()(tensors)
    elif len(tensors) == 1:
        prev = tensors[0]
    else:
        prev = inputs

    return prev


def convert_to_keras_tensor(op, prev):
    if isinstance(op, Dense):
        op.layer = keras.layers.Dense(
            units=op.units,
            activation=op.activation,
            use_bias=op.bias,
            name=op.ID
        )
    elif isinstance(op, Conv2D):
        op.layer = keras.layers.Conv2D(
            filters=op.filters,
            kernel_size=op.kernel,
            strides=op.strides,
            padding=op.padding,
            activation=op.activation,
            use_bias=op.bias,
            name=op.ID
        )
    elif isinstance(op, AvgPooling2x2):
        op.layer = keras.layers.AveragePooling2D(
            pool_size=op.pool_size,
            strides=op.strides,
            padding=op.padding,
            name=op.ID
        )
    elif isinstance(op, MaxPooling2x2):
        op.layer = keras.layers.MaxPooling2D(
            pool_size=op.pool_size,
            strides=op.strides,
            padding=op.padding,
            name=op.ID
        )
    else:
        raise NotImplementedError("Found an unknown op...")

    tensor = op.layer(prev)
    if (isinstance(op, Dense) or isinstance(op, Conv2D)) and op.dropout:
        tensor = keras.layers.Dropout(rate=op.dropout_probability)(tensor)
    return tensor


def build_model(op, inputs):
    # Checking for edge cases:
    if len(op.prev) > 1 and any([p.tensor is None for p in op.prev]):
        return

    # Connecting the Keras tensor operations:
    prev = ensure_correct_tensor_by_op(op, inputs)
    op.tensor = convert_to_keras_tensor(op, prev)

    # Continuing work:
    out = [op.tensor]
    if len(op.next) > 1:
        for next_op in op.next:
            out = build_model(next_op, inputs)
    elif len(op.next) == 1:
        out = build_model(op.next[0], inputs)
    return out


def create_output_tensor(final_tensors, classes):
    tensor = ensure_correct_tensor_by_shape(final_tensors)
    return keras.layers.Dense(units=classes, activation='softmax')(tensor)


def module_to_model(module, input_shape, classes):
    # Preparing module:
    if any([isinstance(op, Module) for op in module.children]):
        module = module.flatten()
    first = module.find_first()
    for op in module.children:
        op.layer = None
        op.tensor = None

    # Build model:
    inputs = keras.layers.Input(shape=input_shape)
    softmax = create_output_tensor(build_model(first, inputs), classes)
    model = keras.models.Model(inputs=[inputs], outputs=[softmax])

    # Clean up module:
    module.clean()
    return model
