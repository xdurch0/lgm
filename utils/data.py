import gzip
import hashlib
import os
import pickle
import re
import shutil
import tarfile
import urllib.request

import tensorflow as tf
import numpy as np
import scipy.io
import librosa


################################################################################
# Downloads etc.
################################################################################
def download_unpack(url, dest_dir):
    """Downloads and (if needed) uncompresses the requested URL.

    Parameters:
        url: Url to download from.
        dest_dir: Stuff will be downloaded in here.

    """
    fname = os.path.split(url)[-1]
    down_dest = os.path.join(dest_dir, fname)
    if url.endswith(".tar.gz"):
        dest_name = down_dest[:-7]
    elif url.endswith(".gz"):
        dest_name = down_dest[:-3]
    else:
        dest_name = down_dest

    if os.path.exists(dest_name):
        print("Final destination {} already exists. "
              "Skipping...".format(dest_name))
        return
    if not os.path.exists(down_dest):
        print("Downloading from {}...".format(url))
        with urllib.request.urlopen(url) as response, open(down_dest, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    else:
        print("Download destination {} already exists.".format(down_dest))

    print("Unpacking...")
    if url.endswith(".tar.gz"):
        with tarfile.open(down_dest, "r") as tar:
            tar.extractall(dest_dir)
    elif url.endswith(".gz"):
        with gzip.open(down_dest, "rb") as f_in:
            with open(down_dest[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def convert_raw_npy(folder):
    """Convert the amazing original MNIST format to numpy arrays.

    Adapted from http://pjreddie.com/projects/mnist-in-csv/.

    """
    def one(subset, n_total):
        img_file = open(os.path.join(folder, subset + "-images-idx3-ubyte"),
                        "rb")
        label_file = open(os.path.join(folder, subset + "-labels-idx1-ubyte"),
                          "rb")

        img_file.read(16)
        label_file.read(8)

        img_array = np.zeros((n_total, 28 * 28), dtype=np.uint8)
        label_array = np.zeros(n_total, dtype=np.uint8)
        for ind_img in range(n_total):
            label_array[ind_img] = ord(label_file.read(1))
            for ind_pix in range(28 * 28):
                img_array[ind_img, ind_pix] = ord(img_file.read(1))
        img_file.close()
        label_file.close()

        if subset == "t10k":
            subset = "test"
        np.save(os.path.join(folder, subset + "_imgs.npy"), img_array)
        np.save(os.path.join(folder, subset + "_lbls.npy"), label_array)

    one("train", 60000)
    one("t10k", 10000)


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
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
    print("Building MNIST TFR...")
    for subset in ["train", "test"]:
        print(subset + "...")
        lbls = np.load(os.path.join(
            array_path, subset + "_lbls.npy")).astype(np.int32)[:, np.newaxis]
        imgs = np.load(os.path.join(
            array_path, subset + "_imgs.npy"))
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
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
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
        target_path: BASE path for TFrecords. Will create two files, do not
                     give file ending here.

    """
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
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
        target_path: BASE path for TFrecords. Will create two files, do not
                     give file ending here.

    """
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
    print("Building CIFAR100 TFR...")

    def one_batch(batch_name):
        with open(os.path.join(pickle_path, batch_name), "rb") as pkl:
            data_dict = pickle.load(pkl, encoding="bytes")
        imgs = np.asarray(data_dict[b"data"])
        lbls = np.asarray(data_dict[b"fine_labels"])
        imgs = imgs.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        lbls = lbls[:, np.newaxis]
        return imgs, lbls

    train_imgs, train_lbls = one_batch("train")
    test_imgs, test_lbls = one_batch("test")

    write_img_label_tfr(target_path + "_train.tfr", train_imgs, train_lbls)
    write_img_label_tfr(target_path + "_test.tfr", test_imgs, test_lbls)


def tfsc_tfr(sc_path, target_path):
    """Build Tensorflow Speech Commands TFRecords.

    Parameters:
        sc_path: Folder to Speech Commands raw files.
        target_path: BASE path for TFrecords. Will create three files do not
                     give file ending here.

    """
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
    print("Building Tensorflow Speech Commands TFR...")

    def which_set(filename, validation_percentage, testing_percentage):
        """Taken from the README."""
        max_num_wavs_per_class = 2 ** 27 - 1  # ~134M
        base_name = os.path.basename(filename)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (max_num_wavs_per_class + 1)) *
                           (100.0 / max_num_wavs_per_class))
        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'testing'
        else:
            result = 'training'
        return result

    exclude = {"_background_noise_"}
    train_audio = []
    train_lbls = []
    test_audio = []
    test_lbls = []
    val_audio = []
    val_lbls = []

    label_ind = -1
    for folder in os.listdir(sc_path):
        base = os.path.join(sc_path, folder)
        if folder in exclude or not os.path.isdir(base):
            continue
        print("Doing {}...".format(folder))
        label_ind += 1
        for file in os.listdir(base):
            audio, _ = librosa.load(os.path.join(base, file), sr=None,
                                    duration=1, dtype=np.float16)
            if len(audio) < 16384:
                audio = np.pad(audio, [0, 16384-len(audio)], "constant")
            dset = which_set(file, 10, 10)
            if dset == "training":
                train_audio.append(audio)
                train_lbls.append([label_ind])
            elif dset == "testing":
                test_audio.append(audio)
                test_lbls.append([label_ind])
            else:
                val_audio.append(audio)
                val_lbls.append([label_ind])

    write_img_label_tfr(target_path + "_train.tfr", train_audio, train_lbls)
    write_img_label_tfr(target_path + "_test.tfr", test_audio, test_lbls)
    write_img_label_tfr(target_path + "_val.tfr", val_audio, val_lbls)


################################################################################
# Generic functions
################################################################################
def write_img_label_tfr(target_path, imgs, lbls):
    """Write a simple image/label dataset to TFRecords.

    Parameters:
        target_path: Path to store the tfrecords file to.
        imgs: Array of images.
        lbls: Array of labels. Should be 2D!! I.e. even if each label just a
              single number (e.g. one-hot index), wrap that as a 1-element
              vector.

    """
    if os.path.exists(target_path):
        print("File {} exists. Skipping creation...".format(target_path))
        return
    with tf.io.TFRecordWriter(target_path) as writer:
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


def parse_img_label_tfr(example_proto, shape, img_dtype=tf.uint8, to01=True,
                        img_key="img", lbl_key="lbl"):
    """Parse function for TFRecords dataset, for data.map().

    Parameters:
        example_proto: protobuf of single tf.Example.
        shape: Shape of the "image" entry in the example to reshape to.
        img_dtype: Dtype of the "image" data.
        to01: If true, image data will be assumed to be currently stored in
              [0, 255] and will be rescaled to [0, 1].
        img_key: Key that holds the "image" entry in the example.
        lbl_key: Key that holds the "label" entry in the example.

    Returns:
        Tuple of image (reshaped and cast to float32), label.

    """
    features = {img_key: tf.io.FixedLenFeature((), tf.string),
                lbl_key: tf.io.FixedLenFeature((), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_img = tf.reshape(
        tf.io.decode_raw(parsed_features[img_key], out_type=img_dtype), shape)
    parsed_img = tf.cast(parsed_img, tf.float32)
    if to01:
        parsed_img = parsed_img / 255.
    return parsed_img, tf.cast(parsed_features[lbl_key], tf.int32)


def parse_nsynth(example_proto):
    features = {"audio": tf.io.FixedLenFeature((4*16000), tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features["audio"]


def tfr_dataset_eager(tfr_paths, batch_size, map_func, shufrep=0):
    """Make dataset from TFRecord file(s).

    Parameters:
        tfr_paths: String or list of strings, paths to TFRecord files.
        batch_size: Desired batch size.
        map_func: Function to map records with.
        shufrep: Int, if given, repeat dataset indefinitely and use this number
                 as shuffle buffer size. Otherwise (default), dataset is neither
                 shuffled nor repeated.

    Returns:
        tf.data.Dataset.

    """
    data = tf.data.TFRecordDataset(tfr_paths)
    if shufrep:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(shufrep))
    data = data.apply(tf.data.experimental.map_and_batch(
        map_func=map_func, batch_size=batch_size))
    data = data.prefetch(1)
    return data
