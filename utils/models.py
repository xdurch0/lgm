import tensorflow as tf
import tensorflow.keras.layers as layers


################################################################################
# Architectures
################################################################################
def gen_fc_mnist(use_bn=False, h_act=tf.nn.leaky_relu, final_act=tf.nn.sigmoid,
                 channels=1):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Dense(256, h_act_internal),
           layers.Dense(512, h_act_internal),
           layers.Dense(1024, h_act_internal),
           layers.Dense(32*32*channels, final_act),
           layers.Reshape((32, 32, channels))]
    if use_bn:
        seq = add_bn(seq, h_act, up_to=-2)
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
                                  activation=final_act)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def gen_conv_mnist_nn(use_bn=False, h_act=tf.nn.leaky_relu,
                      final_act=tf.nn.sigmoid, channels=1):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Reshape((1, 1, -1)),
           layers.Upsampling2D(),
           layers.Conv2D(256, 4, padding="same", activation=h_act_internal),
           layers.Upsampling2D(),
           layers.Conv2D(128, 4, padding="same", activation=h_act_internal),
           layers.Upsampling2D(),
           layers.Conv2D(64, 4, padding="same", activation=h_act_internal),
           layers.Upsampling2D(),
           layers.Conv2D(32, 4, padding="same", activation=h_act_internal),
           layers.Upsampling2D(),
           layers.Conv2D(channels, 4, padding="same", activation=final_act),
           ]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def enc_fc_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu, clip=None):
    h_act_internal = None if use_bn else h_act
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Reshape((-1,)),
           layers.Dense(512, h_act_internal, kernel_constraint=const,
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


def enc_conv_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu, clip=None):
    h_act_internal = None if use_bn else h_act
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Conv2D(32, 3, padding="same", activation=h_act_internal,
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


################################################################################
# Helpers
################################################################################
def add_bn(layer_seq, h_act, up_to=-1):
    """Apply batchnorm to a sequence of layers.

    Apply only to layers with parameters. Layers are assumed to have no
    activation; this is applied here, after batchnorm.

    Parameters:
        layer_seq: List/iterable of keras layers.
        h_act: Activation to apply after batchnorm.
        up_to: Int (usually negative); e.g. if -1, do not apply batchnorm to
               the last layer. If -2, do not apply to the last two layers etc.
               Note that this count includes non-eligible layers, so let's say
               if the last conv layer should not have batchnorm applied to but
               it is followed by a Reshape layer, this needs to be -2.
               TODO this is awful lol.
    """
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


def model_up_to(model, up_to):
    """Applies only the first n layers of a model.

    Parameters:
        model: Keras model with a layers attribute.
        up_to: Int, how many layers of the model should be applied. E.g.
               passing 1 will apply only the first layer (index 0). Basically
               this is the index of the first layer that is excluded.

    Returns:
        Callable that only applies the requested layers.
    """
    def partial_model(inp):
        for ind in range(up_to):
            inp = model.layers[ind](inp)
        return inp

    return partial_model
