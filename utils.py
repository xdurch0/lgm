import tensorflow as tf
import numpy as np


def make_mnist_iter(batch_size, train=True):
    pl_imgs = tf.placeholder(tf.float32, [None, 784])
    pl_lbls = tf.placeholder(tf.int32, [None])
    data = tf.data.Dataset.from_tensor_slices((pl_imgs, pl_lbls))
    if train:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(60000))
    data = data.batch(batch_size)
    return data.make_initializable_iterator(), pl_imgs, pl_lbls


def init_mnist(sess, iterator, pl_imgs, pl_lbls, train=True, normalize=True,
               binarize=False):
    if train:
        imgs = np.load("data/mnist_train_imgs.npy")
        lbls = np.load("data/mnist_train_lbls.npy")
    else:
        imgs = np.load("data/mnist_test_imgs.npy")
        lbls = np.load("data/mnist_test_lbls.npy")
    lbls = lbls.astype(np.int32)
    if normalize:
        imgs = imgs.astype(np.float32) / 255
    if binarize:
        if not normalize:
            raise ValueError("Binarization not implemented for unnormalized "
                             "data.")
        imgs = np.around(imgs)
    sess.run(iterator.initializer, feed_dict={pl_imgs: imgs, pl_lbls: lbls})


def mnist_eager(batch_size, train=True, normalize=True, binarize=False):
    if train:
        imgs = np.load("data/mnist_train_imgs.npy")
        lbls = np.load("data/mnist_train_lbls.npy")
    else:
        imgs = np.load("data/mnist_test_imgs.npy")
        lbls = np.load("data/mnist_test_lbls.npy")
    lbls = lbls.astype(np.int32)
    if normalize:
        imgs = imgs.astype(np.float32) / 255
    if binarize:
        if not normalize:
            raise ValueError("Binarization not implemented for unnormalized "
                             "data.")
        imgs = np.around(imgs)
    data = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    if train:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(60000))
    data = data.batch(batch_size)
    return data


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
# RBM
################################################################################
def repeated_gibbs(init_sample, n_iters, gibbs_update_fn, **guf_kwargs):
    """Repeatedly apply Gibbs updates for a given number of iterations.

    Parameters:
        init_sample: Batch of input samples to start with. Needs to be in the
                     appropriate format for the update function (e.g. tuple of
                     visible/hidden for RBMs).
        n_iters: How many Gibbs updates to do.
        gibbs_update_fn: Function that takes a batch of input samples and
                         computes a new one.
        guf_kwargs: Keyword arguments passed to gibbs_update_fn.

    Returns:
        New batch of input samples.

    """
    iter_dummy = tf.range(n_iters)
    fn = lambda sample, dummy: gibbs_update_fn(sample, **guf_kwargs)

    return tf.foldl(fn, iter_dummy, initializer=init_sample, back_prop=False)


def gibbs_update_brbm(prev_sample, w_vh, b_v, b_h):
    """Gibbs update step for binary RBMs.

    Given an input sample, take a hidden sample and then a new input sample.

    Parameters:
        prev_sample: Tuple of b x d_v tensor and b x d_h tensor: Both batches
                     of input/hidden samples.
        w_vh: Connection matrix of RBM, d_v x d_h.
        b_v: Bias vector for inputs, d_v-dimensional.
        b_h: Bias vector for hidden variables, d_h-dimensional.

    Returns:
        New batch of input/hidden samples as tuple.

    """
    v, _ = prev_sample

    p_h_v = tf.nn.sigmoid(tf.matmul(v, w_vh) + b_h)
    sample_h = tf.distributions.Bernoulli(
        probs=p_h_v, dtype=tf.float32).sample()
    p_v_h = tf.nn.sigmoid(tf.matmul(sample_h, tf.transpose(w_vh)) + b_v)
    sample_v = tf.distributions.Bernoulli(
        probs=p_v_h, dtype=tf.float32).sample()

    return sample_v, sample_h


def energy_rbm(v, h, w_vh, b_v, b_h):
    """Compute energy for an RBM.

    Parameters:
        v: Batch of inputs, b x d_v.
        h: Batch of hidden units, b x d_h.
        w_vh: Connection matrix of RBM, d_v x d_h.
        b_v: Bias vector for inputs, d_v-dimensional.
        b_h = Bias vector for hidden variables, d_h-dimensional.

    Returns:
        b-dimensional vector, energy for each batch element.

    """
    return (-matvec(v, b_v) - matvec(h, b_h) -
            tf.reduce_sum(tf.matmul(v, w_vh) * h, axis=-1))


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
