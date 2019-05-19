import tensorflow as tf
import numpy as np


def matvec(mat, vec):
    """Multiply a matrix with a vector to receive a vector.

    Parameters:
        mat: m x n matrix (2d tensor).
        vec: n-vector (1d tensor of size n).

    returns:
        m-vector.

    """
    return tf.squeeze(tf.matmul(mat, vec[:, tf.newaxis]), axis=-1)


################################################################################
# MMD
################################################################################
def rbf_kernel(x, y, sigma_sqr):
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
        -tf.reduce_mean(tf.math.squared_difference(x_broadcast, y_broadcast),
                        axis=2) / sigma_sqr)


def imq_kernel(x, y, c):
    """It's a different kernel."""
    x_broadcast = x[:, tf.newaxis, :]
    y_broadcast = y[tf.newaxis, :, :]
    return c / (c + tf.reduce_sum(
        tf.math.squared_difference(x_broadcast, y_broadcast)))


def compute_mmd(x, y, kernel_fn, scales):
    """Compute MMD between two batches of vectors.

    Parameters:
        x: n x d tensor of floats.
        y: Like x.
        kernel_fn: Function that takes two batches of n inputs and computes
                   an n x n kernel matrix. It should also take a third input
                   that encodes the scale of the kernel.
        scales: List of scales for the kernel function. One kernel matrix is
                computed per entry and the results are added. Alternatively,
                this can just be a single number.

    Returns:
        Scalar MMD value.

    """
    if isinstance(scales, float) or isinstance(scales, int):
        scales = [scales]

    total = 0
    for sc in scales:
        x_kernel = kernel_fn(x, x, sc)
        y_kernel = kernel_fn(y, y, sc)
        xy_kernel = kernel_fn(x, y, sc)
        total += tf.reduce_mean(x_kernel + y_kernel - 2 * xy_kernel)
    return total


################################################################################
# Parzen windows
################################################################################
def log_density_gaussian(test_data, model_samples, sigma_sqr,
                         include_constant=False):
    """Compute log density estimate for Gaussian window.

    Parameters:
        test_data: b x d tensor, batch of data for which we want the probability.
        model_samples: n x d tensor, data to be used to estimate the density.
        sigma_sqr: Variance of the Gaussian kernel.
        include_constant: If true, add a data- and model-independent (as long as
                          data dimensionality is the same) constant to the log
                          density; necessary to get correct values. If only
                          comparing between different models, you can leave this
                          False.

    Returns:
        batch-size tensor of probabilities for each element in test_data.
    """
    x_broadcast = test_data[:, tf.newaxis, :]
    y_broadcast = model_samples[tf.newaxis, :, :]
    n_samples = tf.cast(tf.shape(model_samples)[0], tf.float32)
    dim = tf.cast(tf.shape(model_samples)[-1], tf.float32)

    inner = tf.exp(-tf.reduce_sum(tf.math.squared_difference(x_broadcast, y_broadcast),
                                  axis=2) / (2 * sigma_sqr))
    constant = 0.5*dim*tf.math.log(2*np.pi) if include_constant else 0
    return (tf.math.log(tf.reduce_sum(inner, axis=-1)) - tf.math.log(n_samples)
            - 0.5*tf.math.log(sigma_sqr)) - constant


def log_density_uniform(test_data, model_samples, width):
    """Compute log density estimate for Uniform window.

    Parameters:
        test_data: b x d tensor, batch of data for which we want the probability.
        model_samples: n x d tensor, data to be used to estimate the density.
        width: Width of the uniform kernel.

    Returns:
        batch-size tensor of probabilities for each element in test_data.
    """
    x_broadcast = test_data[:, tf.newaxis, :]
    y_broadcast = model_samples[tf.newaxis, :, :]
    n_samples = tf.cast(tf.shape(model_samples)[0], tf.float32)
    dim = tf.cast(tf.shape(model_samples)[-1], tf.float32)

    max_dists = tf.norm(tf.abs(x_broadcast - y_broadcast), ord=np.inf, axis=-1)
    in_vals = tf.fill(tf.shape(max_dists), 1.)
    out_vals = tf.fill(tf.shape(max_dists), 0.)
    indic = tf.where(tf.less(max_dists, width/2), in_vals, out_vals)
    return (tf.math.log(tf.reduce_sum(indic, axis=-1)) - tf.math.log(n_samples)
            - dim*tf.math.log(width))
