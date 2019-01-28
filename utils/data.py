import os

import tensorflow as tf
import numpy as np
import scipy.io


################################################################################
# MNIST
################################################################################
def make_mnist_iter(batch_size, shufrep=True, to32=True):
    """Make initializable iterator for MNIST.

    Use placeholders to avoid having the dataset as a constant in the graph.

    Parameters:
        batch_size: Desired batch size.
        shufrep: If true, shuffle and repeat indefinitely.
        to32: If true, pad images to 32x32.

    Returns:
        Initializable iterator; image placeholder; label placeholder.

    """
    img_shape = [None, 32, 32, 1] if to32 else [None, 28, 28, 1]
    pl_imgs = tf.placeholder(tf.float32, img_shape)
    pl_lbls = tf.placeholder(tf.int32, [None])
    data = tf.data.Dataset.from_tensor_slices((pl_imgs, pl_lbls))
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(60000))
    data = data.batch(batch_size)
    return data.make_initializable_iterator(), pl_imgs, pl_lbls


def init_mnist(base_path, which, sess, iterator, pl_imgs, pl_lbls,
               normalize=True, binarize=False, to32=True):
    """Initialize placeholder MNIST dataset.

    Parameters:
        base_path: Path to folder with numpy arrays.
        which: String giving which subset to take, or list of strings to take
               multiple subsets. Valid: train, test.
        sess: Tensorflow session.
        iterator: As gotten by make_mnist_iter.
        pl_imgs: Same.
        pl_lbls: Same.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        to32: If true, pad images to 32x32.

    """
    imgs, lbls = preprocess_mnist(base_path, which, normalize, binarize, to32)
    sess.run(iterator.initializer, feed_dict={pl_imgs: imgs, pl_lbls: lbls})


def mnist_eager(base_path, which, batch_size, shufrep=True, normalize=True,
                binarize=False, to32=True):
    """Dataset for eager execution.

    Parameters:
        base_path: Path to folder with numpy arrays.
        which: String giving which subset to take, or list of strings to take
               multiple subsets. Valid: train, test.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        batch_size: Desired batch size.
        shufrep: If true, shuffle and repeat indefinitely.
        to32: If true, pad images to 32x32.

    Returns:
          TF Dataset.

    """
    imgs, lbls = preprocess_mnist(base_path, which, normalize, binarize, to32)
    data = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(len(imgs)))
    data = data.batch(batch_size)
    return data


def preprocess_mnist(base_path, which, normalize=True, binarize=False,
                     to32=True):
    """Preprocess numpy MNIST arrays.

    Parameters:
        base_path: Path to folder with numpy arrays.
        which: String giving which subset to take, or list of strings to take
               multiple subsets. Valid: train, test.
        normalize: If true, normalize data to [0, 1].
        binarize: If true, round data to only 0 and 1.
        to32: If true, pad images to 32x32.

    Returns:
        Array of images, array of labels.

    """
    if isinstance(which, str):
        which = [which]

    imgs = np.empty((0, 784), dtype=np.uint8)
    lbls = np.empty((0,), dtype=np.int32)
    for subset in which:
        imgs = np.concatenate(
            (imgs,
             np.load(os.path.join(base_path, "mnist_" + subset + "_imgs.npy"))))
        lbls = np.concatenate(
            (lbls,
             np.load(os.path.join(base_path, "mnist_" + subset +
                                  "_lbls.npy")).astype(np.int32)))

    imgs = imgs.reshape((-1, 28, 28, 1))
    if to32:
        imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")
    if normalize:
        imgs = imgs.astype(np.float32) / 255
    if binarize:
        if not normalize:
            raise ValueError("Binarization not implemented for unnormalized "
                             "data.")
        imgs = np.around(imgs)
    return imgs, lbls


################################################################################
# SVHN
################################################################################
def svhn_eager(base_path, which, batch_size, normalize=True, binarize=False,
               shufrep=True):
    if isinstance(which, str):
        which = [which]
    imgs = np.empty((0, 32, 32, 3), dtype=np.uint8)
    lbls = np.empty((0, 1), dtype=np.int32)
    for subset in which:
        matdict = scipy.io.loadmat(os.path.join(base_path,
                                                subset + "_32x32.mat"))
        imgs = np.concatenate((imgs, np.transpose(matdict["X"], [3, 0, 1, 2])))
        lbls = np.concatenate((lbls, matdict["y"].astype(np.int32)))

    if normalize:
        imgs = imgs.astype(np.float32) / 255
    if binarize:
        if not normalize:
            raise ValueError("Binarization not implemented for unnormalized "
                             "data.")
        imgs = np.around(imgs)

    data = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(len(imgs)))
    data = data.batch(batch_size)
    return data


################################################################################
# TFRecords :(
################################################################################
def mnist_tfr(array_path, target_path, to32=True):
    """Build MNIST TFRecords.

    Parameters:
        array_path: Path to folder where the arrays are stored.
        target_path: BASE path for TFrecords. Will create two files, do not give
                     file ending here.
        to32: If true, pad images to 32x32.

    """
    for subset in ["train", "test"]:
        lbls = np.load(os.path.join(
            array_path, "mnist_" + subset + "_lbls.npy")).astype(np.int32)[:, np.newaxis]
        imgs = np.load(os.path.join(
            array_path, "mnist_" + subset + "_imgs.npy")).astype(np.float32) / 255.
        imgs = imgs.reshape((-1, 28, 28, 1))
        if to32:
            imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")
        write_img_label_tfr(target_path + "_" + subset + ".tfr", imgs, lbls)


def svhn_tfr(mat_path, target_path):
    """Build SVHN TFRecords.

    Parameters:
        mat_path: Path to folder where the mat files are stored.
        target_path: BASE path for TFrecords. Will create three files, do not
                     give file ending here.

    """
    for subset in ["train", "extra", "test"]:
        matdict = scipy.io.loadmat(os.path.join(mat_path,
                                                subset + "_32x32.mat"))
        imgs = np.transpose(matdict["X"], [3, 0, 1, 2]).astype(np.float32) / 255.
        lbls = matdict["y"].astype(np.int32)
        write_img_label_tfr(target_path + "_" + subset + ".tfr", imgs, lbls)


def write_img_label_tfr(target_path, imgs, lbls):
    """Write a simple image/label dataset to TFRecords.

    Parameters:
        target_path: Path to store the tfrecords file to.
        imgs: Array of images. Should be float32.
        lbls: Array of labels. Should be 2D!! I.e. even if it's just a single
              number (one-hot index), wrap that as a 1-element vector.

    """
    with tf.python_io.TFRecordWriter(target_path) as writer:
        for img, lbl in zip(imgs, lbls):
            tfex = tf.train.Example(features=tf.train.Features(
                feature={"img": float_feature(img.flatten()),
                         "lbl": int64_feature(lbl)}))
            writer.write(tfex.SerializeToString())


def float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))


def int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))


def parse_img_label_tfr(example_proto, shape):
    features = {"img": tf.FixedLenFeature((np.prod(shape),), tf.float32),
                "lbl": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return (tf.reshape(parsed_features["img"], shape) ,
            tf.cast(parsed_features["lbl"], tf.int32))


def tfr_dataset_eager(tfr_paths, batch_size, map_func, shufrep=0, cache=False):
    data = tf.data.TFRecordDataset(tfr_paths)
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(shufrep))
    data = data.apply(tf.data.experimental.map_and_batch(
        map_func=map_func, batch_size=batch_size))
    data = data.prefetch(1)
    if cache:
        data = data.cache()
    return data