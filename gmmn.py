import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from mnist import MNISTDataset


# data
batch_size = 128
mnistd = MNISTDataset("data", batch_size)
train_steps = 1500
noise = 100


def gen(shape_dummy):
    """Sampler from random noise."""
    with tf.variable_scope("generator"):
        noise = tf.random_uniform(shape_dummy, minval=-1, maxval=1)
        layers = [256, 512, 1024]
        out = noise
        for n_h in layers:
            out = tf.layers.dense(out, n_h, activation=tf.nn.relu)
        out = tf.layers.dense(out, 784, activation=tf.nn.sigmoid)
        return out


def compute_kernel(x, y, sigma_sqr):
    """Compute pairwise similarity measure between two batches of vectors.
    Parameters:
        x: n x d tensor of floats.
        y: Like x.
        sigma_sqr: Variance for the Gaussian kernel.
    Returns:
        n x n tensor where element i, j contains similarity between element i
        in x and element j in y.
    """
    x_broadcast = x[:, tf.newaxis, :]
    y_broadcast = y[tf.newaxis, :, :]
    return tf.exp(
        -tf.reduce_mean(tf.squared_difference(x_broadcast, y_broadcast),
                        axis=2) / sigma_sqr)


def compute_mmd(x, y, sigma_sqr=None):
    """Compute MMD between two batches of vectors.
    Parameters:
        x: n x d tensor of floats.
        y: Like x.
        sigma_sqr: Variance for the Gaussian kernel.
    Returns:
        Scalar MMD value.
    """
    if sigma_sqr is None:
        sigma_sqr = tf.cast(x.shape.as_list()[1], tf.float32)
    x_kernel = compute_kernel(x, x, sigma_sqr)
    y_kernel = compute_kernel(y, y, sigma_sqr)
    xy_kernel = compute_kernel(x, y, sigma_sqr)
    return tf.reduce_mean(x_kernel + y_kernel - 2 * xy_kernel)


inp_pl = tf.placeholder(tf.float32, [None, 784])
generated = gen(tf.shape(inp_pl))

loss = compute_mmd(inp_pl, generated)

opt = tf.train.AdamOptimizer()
grads_vars = opt.compute_gradients(loss)
take_step = opt.apply_gradients(grads_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(train_steps):
        _,  mmd = sess.run([take_step, loss], feed_dict={inp_pl: mnistd.next_batch()[0]})

        if not step % 50:
            print("Step", step)
            print("Loss", mmd)
