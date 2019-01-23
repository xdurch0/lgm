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
        sigma_sqr: Variance for the Gaussian kernel. Can be None to choose
                   default value based on number of dimensions, a scalar, or
                   a list of scalars to add multiple kernels.

    Returns:
        Scalar MMD value.
    """
    if sigma_sqr is None:
        sigma_sqr = tf.cast(x.shape.as_list()[1], tf.float32)
    elif type(sigma_sqr) is float or type(sigma_sqr) is int:
        sigma_sqr = [sigma_sqr]
    total = 0
    for sig in sigma_sqr:
        x_kernel = compute_kernel(x, x, sig)
        y_kernel = compute_kernel(y, y, sig)
        xy_kernel = compute_kernel(x, y, sig)
        total += tf.reduce_mean(x_kernel + y_kernel - 2 * xy_kernel)
    return total
