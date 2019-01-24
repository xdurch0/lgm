import tensorflow as tf
import tensorflow.keras.layers as layers


def gen_fc_mnist(use_bn=False, h_act=tf.nn.leaky_relu, final_act=tf.nn.sigmoid):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Dense(256, h_act_internal),
           layers.Dense(512, h_act_internal),
           layers.Dense(1024, h_act_internal),
           layers.Dense(1024, final_act)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def gen_conv_mnist(use_bn=False, h_act=tf.nn.leaky_relu,
                   final_act=tf.nn.sigmoid):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Reshape((1, 1, -1)),
           layers.Conv2DTranspose(64, 3, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(64, 3, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(32, 3, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(32, 3, 2, padding="same",
                                  activation=h_act_internal),
           layers.Conv2DTranspose(1, 3, 2, padding="same",
                                  activation=final_act),
           layers.Flatten()]
    if use_bn:
        seq = add_bn(seq, h_act, up_to=-2)
    return tf.keras.Sequential(seq)


def enc_fc_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Dense(512, h_act_internal),
           layers.Dense(256, h_act_internal),
           layers.Dense(128, h_act_internal),
           layers.Dense(final_dim)]
    if use_bn:
        seq = add_bn(seq, h_act)
    return tf.keras.Sequential(seq)


def enc_conv_mnist(final_dim, use_bn=False, h_act=tf.nn.leaky_relu):
    h_act_internal = None if use_bn else h_act
    seq = [layers.Reshape((32, 32, 1)),
           layers.Conv2D(32, 3, padding="same", activation=h_act_internal),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(32, 3, padding="same", activation=h_act_internal),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(32, 3, padding="same", activation=h_act_internal),
           layers.AveragePooling2D(padding="same"),
           layers.Conv2D(32, 3, padding="same", activation=h_act_internal),
           layers.AveragePooling2D(padding="same"),
           layers.Flatten(),
           layers.Dense(final_dim)]
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
    return (isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv2D)
            or isinstance(layer, layers.Conv2DTranspose))