import os

import tensorflow as tf
tf.enable_eager_execution()

from utils.data import mnist_tfr, svhn_tfr, cifar10_tfr, cifar100_tfr


BASE_RAW = "/cache"
BASE_TFRS = "/cache/tfrs"
mnist_tfr(os.path.join(BASE_RAW, "mnist"), os.path.join(BASE_TFRS, "mnist"))
mnist_tfr(os.path.join(BASE_RAW, "fashion"), os.path.join(BASE_TFRS, "fashion"))
svhn_tfr(os.path.join(BASE_RAW, "svhn"), os.path.join(BASE_TFRS, "svhn"))
cifar10_tfr(os.path.join(BASE_RAW, "cifar-10-batches-py"), os.path.join(BASE_TFRS, "cifar10"))
cifar100_tfr(os.path.join(BASE_RAW, "cifar-100-python"), os.path.join(BASE_TFRS, "cifar100"))
