import tensorflow as tf


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
