import os
import pickle

import tensorflow as tf
import numpy as np
import scipy.io


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
    print("Building MNIST TFR...")
    for subset in ["train", "test"]:
        print(subset + "...")
        lbls = np.load(os.path.join(
            array_path, "mnist_" + subset + "_lbls.npy")).astype(np.int32)[:, np.newaxis]
        imgs = np.load(os.path.join(
            array_path, "mnist_" + subset + "_imgs.npy"))
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
    print("Building SVHN TFR...")
    for subset in ["train", "extra", "test"]:
        print(subset + "...")
        matdict = scipy.io.loadmat(os.path.join(mat_path,
                                                subset + "_32x32.mat"))
        imgs = np.transpose(matdict["X"], [3, 0, 1, 2])
        lbls = matdict["y"].astype(np.int32)
        write_img_label_tfr(target_path + "_" + subset + ".tfr", imgs, lbls)


def cifar10_tfr(pickle_path, target_path):
    """Build CIFAR10 TFRecords.

    Parameters:
        pickle_path: Path to folder where the pickle files are stored.
        target_path: BASE path for TFrecords. Will create three files, do not
                     give file ending here.

    """
    print("Building CIFAR10 TFR...")

    def one_batch(batch_name):
        with open(os.path.join(pickle_path, batch_name), "rb") as pkl:
            data_dict = pickle.load(pkl, encoding="bytes")
        imgs = np.asarray(data_dict[b"data"])
        lbls = np.asarray(data_dict[b"labels"])
        imgs = imgs.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        lbls = lbls[:, np.newaxis]
        return imgs, lbls

    train_imgs, train_lbls = zip(*[one_batch("data_batch_" + str(num))
                                   for num in range(1, 6)])
    train_imgs = np.concatenate(train_imgs)
    train_lbls = np.concatenate(train_lbls)

    test_imgs, test_lbls = one_batch("test_batch")

    write_img_label_tfr(target_path + "_train.tfr", train_imgs, train_lbls)
    write_img_label_tfr(target_path + "_test.tfr", test_imgs, test_lbls)


def cifar100_tfr(pickle_path, target_path):
    """Build CIFAR100 TFRecords.

    Parameters:
        pickle_path: Path to folder where the pickle files are stored.
        target_path: BASE path for TFrecords. Will create three files, do not
                     give file ending here.

    """
    print("Building CIFAR100 TFR...")

    def one_batch(batch_name):
        with open(os.path.join(pickle_path, batch_name), "rb") as pkl:
            data_dict = pickle.load(pkl, encoding="bytes")
        imgs = np.asarray(data_dict[b"data"])
        lbls = np.asarray(data_dict[b"labels"])
        imgs = imgs.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        lbls = lbls[:, np.newaxis]
        return imgs, lbls

    train_imgs, train_lbls = one_batch("train")
    test_imgs, test_lbls = one_batch("test")

    write_img_label_tfr(target_path + "_train.tfr", train_imgs, train_lbls)
    write_img_label_tfr(target_path + "_test.tfr", test_imgs, test_lbls)


def write_img_label_tfr(target_path, imgs, lbls):
    """Write a simple image/label dataset to TFRecords.

    Parameters:
        target_path: Path to store the tfrecords file to.
        imgs: Array of images. We expect this to be uint8 for more compact
              storage.
        lbls: Array of labels. Should be 2D!! I.e. even if each label just a
              single number (e.g. one-hot index), wrap that as a 1-element
              vector.

    """
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
    with tf.python_io.TFRecordWriter(target_path) as writer:
        for img, lbl in zip(imgs, lbls):
            tfex = tf.train.Example(features=tf.train.Features(
                feature={"img": bytes_feature(img.tobytes()),
                         "lbl": int64_feature(lbl)}))
            writer.write(tfex.SerializeToString())


def bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def float_feature(val):
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))


def int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))


def parse_img_label_tfr(example_proto, shape, to01=True):
    features = {"img": tf.FixedLenFeature((), tf.string),
                "lbl": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_img = tf.reshape(
        tf.decode_raw(parsed_features["img"], out_type=tf.uint8), shape)
    if to01:
        parsed_img = tf.to_float(parsed_img) / 255.
    return parsed_img, tf.cast(parsed_features["lbl"], tf.int32)


def tfr_dataset_eager(tfr_paths, batch_size, map_func, shufrep=0):
    data = tf.data.TFRecordDataset(tfr_paths)
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(shufrep))
    data = data.apply(tf.data.experimental.map_and_batch(
        map_func=map_func, batch_size=batch_size))
    data = data.prefetch(1)
    return data