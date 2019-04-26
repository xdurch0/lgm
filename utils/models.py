import tensorflow as tf
import tensorflow.keras.layers as layers


################################################################################
# Architectures
################################################################################
def gen_fc_mnist(use_bn=True, h_act=layers.LeakyReLU, final_act=tf.nn.sigmoid,
                 norm=layers.BatchNormalization, channels=1):
    seq = [layers.Dense(256, use_bias=not use_bn),
           h_act(),
           layers.Dense(512, use_bias=not use_bn),
           h_act(),
           layers.Dense(1024, use_bias=not use_bn),
           h_act(),
           layers.Dense(32*32*channels, final_act),
           layers.Reshape((32, 32, channels))]
    if use_bn:
        seq = add_norm(seq, up_to=-2, norm=norm)
    return tf.keras.Sequential(seq)


def gen_conv_mnist(use_bn=True, h_act=layers.LeakyReLU, final_act=tf.nn.sigmoid,
                   norm=layers.BatchNormalization, channels=1):
    seq = [layers.Lambda(lambda x: x[:, tf.newaxis, tf.newaxis, :]),
           layers.Conv2DTranspose(256, 4, 2, padding="same", use_bias=not use_bn),
           h_act(),
           layers.Conv2DTranspose(128, 4, 2, padding="same", use_bias=not use_bn),
           h_act(),
           layers.Conv2DTranspose(64, 4, 2, padding="same", use_bias=not use_bn),
           h_act(),
           layers.Conv2DTranspose(32, 4, 2, padding="same", use_bias=not use_bn),
           h_act(),
           layers.Conv2DTranspose(channels, 4, 2, padding="same",
                                  activation=final_act)]
    if use_bn:
        seq = add_norm(seq, norm=norm)
    return tf.keras.Sequential(seq)


def gen_conv_mnist_nn(use_bn=True, h_act=layers.LeakyReLU,
                      final_act=tf.nn.sigmoid, norm=layers.BatchNormalization,
                      channels=1):
    seq = [layers.Lambda(lambda x: x[:, tf.newaxis, tf.newaxis, :]),
           layers.UpSampling2D(),
           layers.Conv2D(256, 4, padding="same", use_bias=not use_bn),
           h_act(),
           layers.UpSampling2D(),
           layers.Conv2D(128, 4, padding="same", use_bias=not use_bn),
           h_act(),
           layers.UpSampling2D(),
           layers.Conv2D(64, 4, padding="same", use_bias=not use_bn),
           h_act(),
           layers.UpSampling2D(),
           layers.Conv2D(32, 4, padding="same", use_bias=not use_bn),
           h_act(),
           layers.UpSampling2D(),
           layers.Conv2D(channels, 4, padding="same", activation=final_act)]
    if use_bn:
        seq = add_norm(seq, norm=norm)
    return tf.keras.Sequential(seq)


def enc_fc_mnist(final_dim, use_bn=True, h_act=layers.LeakyReLU, clip=None,
                 norm=layers.BatchNormalization):
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Flatten(),
           layers.Dense(512, kernel_constraint=const, bias_constraint=const,
                        use_bias=not use_bn),
           h_act(),
           layers.Dense(256, kernel_constraint=const, bias_constraint=const,
                        use_bias=not use_bn),
           h_act(),
           layers.Dense(128, kernel_constraint=const, bias_constraint=const,
                        use_bias=not use_bn),
           h_act(),
           layers.Dense(final_dim, kernel_constraint=const,
                        bias_constraint=const)]
    if use_bn:
        seq = add_norm(seq, norm=norm)
    return tf.keras.Sequential(seq)


def enc_conv_mnist(final_dim, use_bn=True, h_act=layers.LeakyReLU, clip=None,
                   norm=layers.BatchNormalization):
    const = (lambda v: tf.clip_by_value(v, -clip, clip)) if clip else None
    seq = [layers.Conv2D(32, 4, padding="same", kernel_constraint=const,
                         bias_constraint=const, use_bias=not use_bn),
           h_act(),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(64, 4, padding="same", kernel_constraint=const,
                         bias_constraint=const, use_bias=not use_bn),
           h_act(),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(128, 4, padding="same", kernel_constraint=const,
                         bias_constraint=const, use_bias=not use_bn),
           h_act(),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(256, 4, padding="same", kernel_constraint=const,
                         bias_constraint=const, use_bias=not use_bn),
           h_act(),
           layers.AveragePooling2D(padding="same"),
           layers.Flatten(),
           layers.Dense(final_dim, kernel_constraint=const,
                        bias_constraint=const, use_bias=not use_bn)]
    if use_bn:
        seq = add_norm(seq, norm=norm)
    return tf.keras.Sequential(seq)


################################################################################
# Helpers
################################################################################
def add_norm(layer_seq, up_to=-1, norm=layers.BatchNormalization):
    """Apply normalization to a sequence of layers.

    Apply only to layers with parameters.

    Parameters:
        layer_seq: List/iterable of keras layers.
        up_to: Int (usually negative); e.g. if -1, do not apply norm to
               the last layer. If -2, do not apply to the last two layers etc.
               Note that this count includes non-eligible layers, so let's say
               if the last conv layer should not have norm applied to but
               it is followed by a Reshape layer, this needs to be -2.
               TODO this is awful lol.
        norm: What kind of normalization to apply. Should be the constructor of
              a Keras layer such as BatchNormalization (default).
    """
    seq_bn = []
    for layer in layer_seq[:up_to]:
        seq_bn.append(layer)
        if deserves_norm(layer):
            seq_bn.append(norm())
    seq_bn += layer_seq[up_to:]
    return seq_bn


def deserves_norm(layer):
    """Check whether a layer should be followed by (batch)normalization.

    To be extended as needed.
    """
    return (isinstance(layer, layers.Dense) or
            isinstance(layer, layers.Conv2D) or
            isinstance(layer, layers.Conv2DTranspose) or
            isinstance(layer, layers.Conv1D))


def wrap_sigmoid(model):
    """Appends sigmoid activation function to a model.

    Useful for models that output logits e.g. for cross-entropy loss, but later
    you want to look at the probabilities.

    Parameters:
        model: Callable, e.g. Keras model, taking a single argument.

    Returns:
        Callable that applies model and then a sigmoid to the output. Note that
        this is not a Keras model anymore, just a lambda function.
    """
    return lambda x: tf.nn.sigmoid(model(x))


def model_up_to(model, up_to):
    """Applies only the first n layers of a model.

    Parameters:
        model: Keras model with a layers attribute (Sequential).
        up_to: Int, how many layers of the model should be applied. E.g.
               passing 1 will apply only the first layer (index 0). Basically
               this is the index of the first layer that is excluded.

    Returns:
        Keras model (callable) that only applies the requested layers.
    """
    return tf.keras.Sequential(model.layers[:up_to])
