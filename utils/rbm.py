import tensorflow as tf
import tensorflow_probability as tfp

from utils.math import matvec


def repeated_gibbs(init_sample, n_iters, gibbs_update_fn, return_all=False,
                   **guf_kwargs):
    """Repeatedly apply Gibbs updates for a given number of iterations.

    Parameters:
        init_sample: Batch of input samples to start with. Needs to be in the
                     appropriate format for the update function (e.g. tuple of
                     visible/hidden for RBMs).
        n_iters: How many Gibbs updates to do.
        gibbs_update_fn: Function that takes a batch of input samples and
                         computes a new one.
        return_all: If true, return all samples, not just the last one
        guf_kwargs: Keyword arguments passed to gibbs_update_fn.

    Returns:
        New batch of input samples.

    """
    iter_dummy = tf.range(n_iters)

    def fn(sample, _): return gibbs_update_fn(sample, **guf_kwargs)

    loop = tf.scan if return_all else tf.foldl
    return loop(fn, iter_dummy, initializer=init_sample, back_prop=False)


def repeated_gibbs_python(init_sample, n_iters, gibbs_update_fn,
                          return_all=False, **guf_kwargs):
    """Included for pedagogical reasons. ;)"""
    if return_all:
        vis_samps = tf.TensorArray(tf.float32, size=n_iters+1)
        hid_samps = tf.TensorArray(tf.float32, size=n_iters+1)
        vis_samps = vis_samps.write(0, init_sample[0])
        hid_samps = hid_samps.write(0, init_sample[1])
    sample = init_sample
    for ind in tf.range(n_iters):
        sample = gibbs_update_fn(sample, **guf_kwargs)
        if return_all:
            vis_samps = vis_samps.write(ind + 1, sample[0])
            hid_samps = hid_samps.write(ind + 1, sample[1])
    return (vis_samps.stack(), hid_samps.stack()) if return_all else sample


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
    sample_h = tfp.distributions.Bernoulli(
        probs=p_h_v, dtype=tf.float32).sample()
    p_v_h = tf.nn.sigmoid(tf.matmul(sample_h, tf.transpose(w_vh)) + b_v)
    sample_v = tfp.distributions.Bernoulli(
        probs=p_v_h, dtype=tf.float32).sample()

    return sample_v, sample_h


def energy_rbm(v, h, w_vh, b_v, b_h):
    """Compute energy for an RBM.

    Parameters:
        v: Batch of inputs, b x d_v.
        h: Batch of hidden units, b x d_h.
        w_vh: Connection matrix of RBM, d_v x d_h.
        b_v: Bias vector for inputs, d_v-dimensional.
        b_h: Bias vector for hidden variables, d_h-dimensional.

    Returns:
        b-dimensional vector, energy for each batch element.

    """
    return (-matvec(v, b_v) - matvec(h, b_h) -
            tf.reduce_sum(tf.matmul(v, w_vh) * h, axis=-1))
