import tensorflow as tf
import tensorflow.keras.layers as layers


def gen_fc_mnist(use_bn=False, h_act=tf.nn.leaky_relu, final_act=tf.nn.sigmoid,
                 channels=1):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Dense(256, h_act_internal),
           layers.Dense(512, h_act_internal),
           layers.Dense(1024, h_act_internal),
           layers.Dense(32*32*channels, final_act)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def gen_conv_mnist(use_bn=False, h_act=tf.nn.leaky_relu,
                   final_act=tf.nn.sigmoid, channels=1):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Reshape((1, 1, -1)),
           layers.Conv2DTranspose(256, 4, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(128, 4, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(64, 4, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(32, 4, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(channels, 4, 2, padding="same",
                                  activation=final_act),
           layers.Flatten()]
    if use_bn:
        seq = add_bn(seq, h_act, up_to=-2)
    return tf.keras.Sequential(seq)


def enc_fc_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu, clip=None):
    h_act_internal = None if use_bn else h_act
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Dense(512, h_act_internal, kernel_constraint=const,
                        bias_constraint=const),
           layers.Dense(256, h_act_internal, kernel_constraint=const,
                        bias_constraint=const),
           layers.Dense(128, h_act_internal, kernel_constraint=const,
                        bias_constraint=const),
           layers.Dense(final_dim, kernel_constraint=const,
                        bias_constraint=const)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def enc_conv_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu, channels=1,
                   clip=None):
    h_act_internal = None if use_bn else h_act
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Reshape((32, 32, channels)),
           layers.Conv2D(32, 3, padding="same", activation=h_act_internal,
                         kernel_constraint=const, bias_constraint=const),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(64, 3, padding="same", activation=h_act_internal,
                         kernel_constraint=const, bias_constraint=const),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(128, 3, padding="same", activation=h_act_internal,
                         kernel_constraint=const, bias_constraint=const),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(256, 3, padding="same", activation=h_act_internal,
                         kernel_constraint=const, bias_constraint=const),
           layers.AveragePooling2D(padding="same"),
           layers.Flatten(),
           layers.Dense(final_dim, kernel_constraint=const,
                        bias_constraint=const)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def add_bn(layer_seq, h_act, up_to=-1):
    seq_bn = []
    for layer in layer_seq[:up_to]:
        seq_bn.append(layer)
        if deserves_batchnorm(layer):
            seq_bn.append(layers.BatchNormalization())
            seq_bn.append(layers.Lambda(h_act))
    seq_bn += layer_seq[up_to:]
    return seq_bn


def deserves_batchnorm(layer):
    return (isinstance(layer, layers.Dense) or
            isinstance(layer, layers.Conv2D) or
            isinstance(layer, layers.Conv2DTranspose))


def wrap_sigmoid(model):
    """Appends sigmoid activation function to a model.

    Useful for models that output logits e.g. for cross-entropy loss, but later
    you want to look at the probabilities.

    Parameters:
        model: Callable, e.g. Keras model, taking a single argument.

    Returns:
        Callable that applies model and then a sigmoid to the output.
    """
    return lambda x: tf.nn.sigmoid(model(x))
