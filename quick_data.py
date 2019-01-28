import numpy as np
from matplotlib import pyplot as plt
import scipy.io

import tensorflow as tf
tf.enable_eager_execution()

from utils.data import mnist_tfr, svhn_tfr


mnist_tfr("/cache/mnist", "/cache/tfrs/mnist")
mnist_tfr("/cache/fashion", "/cache/tfrs/fashion")
svhn_tfr("/cache/svhn", "/cache/tfrs/svhn")
