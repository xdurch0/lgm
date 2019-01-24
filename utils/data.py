import tensorflow as tf
import numpy as np


def make_mnist_iter(batch_size, train=True, to32=True):
    """Make initializable iterator for MNIST.

    Use placeholders to avoid having the dataset as a constant in the graph.

    Parameters:
        batch_size: Desired batch size.
        train: If true, shuffle and repeat indefinitely.
        to32: If true, pad images to 32x32.

    Returns:
        Initializable iterator; image placeholder; label placeholder.

    """
    pl_imgs = tf.placeholder(tf.float32, [None, 32*32 if to32 else 28*28])
    pl_lbls = tf.placeholder(tf.int32, [None])
    data = tf.data.Dataset.from_tensor_slices((pl_imgs, pl_lbls))
    if train:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(60000))
    data = data.batch(batch_size)
    return data.make_initializable_iterator(), pl_imgs, pl_lbls


def init_mnist(base_path, sess, iterator, pl_imgs, pl_lbls, normalize=True,
               binarize=False, to32=True):
    """Initialize placeholder MNIST dataset.

    Parameters:
        base_path: Path to  numpy arrays with the data.
        sess: Tensorflow session.
        iterator: As gotten by make_mnist_iter.
        pl_imgs: Same.
        pl_lbls: Same.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        to32: If true, pad images to 32x32.

    """
    imgs, lbls = preprocess_mnist(base_path, normalize, binarize, to32)
    sess.run(iterator.initializer, feed_dict={pl_imgs: imgs, pl_lbls: lbls})


def mnist_eager(base_path, batch_size, train=True, normalize=True,
                binarize=False, to32=True):
    """Dataset for eager execution.

    Parameters:
        base_path: Path to  numpy arrays with the data.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        batch_size: Desired batch size.
        train: If true, shuffle and repeat indefinitely.
        to32: If true, pad images to 32x32.

    Returns:
          TF Dataset.

    """
    imgs, lbls = preprocess_mnist(base_path, normalize, binarize, to32)
    data = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    if train:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(60000))
    data = data.batch(batch_size)
    return data


def preprocess_mnist(base_path, normalize=True, binarize=False, to32=True):
    """Preprocess numpy MNIST arrays.

    Parameters:
        base_path: Path to  numpy arrays with the data.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        to32: If true, pad images to 32x32.

    Returns:
        Array of images, array of labels.

    """
    imgs = np.load(base_path + "_imgs.npy")
    lbls = np.load(base_path + "_lbls.npy")
    lbls = lbls.astype(np.int32)
    if to32:
        imgs = imgs.reshape((-1, 28, 28))
        imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), "constant")
        imgs = imgs.reshape((-1, 32*32))
    if normalize:
        imgs = imgs.astype(np.float32) / 255
    if binarize:
        if not normalize:
            raise ValueError("Binarization not implemented for unnormalized "
                             "data.")
        imgs = np.around(imgs)
    return imgs, lbls
