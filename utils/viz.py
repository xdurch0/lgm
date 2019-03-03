import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def random_samples(model, noise_fn, num=10):
    """Generate and plot some random samples.

    Parameters:
        model: Callable that produces images, e.g. Keras Sequential.
        noise_fn: Callable that produces a noise batch, should take (only) batch
                  size as argument.
        num: How many samples to plot.

    """
    samples = model(noise_fn(num)).numpy()
    for sample in samples:
        imshow(sample)


def random_sample_grid(model, noise_fn, grid_dims=(7, 7), show=True):
    """Construct a grid of some random samples in a single image.

    Parameters:
        model: Callable that produces images, e.g. Keras Sequential.
        noise_fn: Callable that produces a noise batch, should take (only) batch
                  size as argument.
        grid_dims: Desired grid dimensions, also specifying how many samples to
              generate.
        show: If true, plot the grid.

    Returns:
        Numpy array with the grid.

    """
    if len(grid_dims) != 2:
        raise ValueError("Grid dimension needs to be 2D.")
    to_gen = np.prod(grid_dims)
    samples = model(noise_fn(to_gen)).numpy()

    grid = img_grid_npy(samples, grid_dims[0], grid_dims[1], normalize=False)
    if show:
        imshow(grid)
    return grid


def interpolate(source, target, granularity=20, gen=None, method="linear"):
    """Interpolate between two images, showing intermediate results.

    Parameters:
        source: Starting point.
        target: End point.
        granularity: How many steps to take from source to target.
        gen: If given, interpolations are assumed to be codes that need to be
             mapped to image space first; gen should be a callable that does
             this.
        method: What interpolation method to use. Either simple linear
                interpolation, or 'slerp' to go along a great circle.
    """
    for mixture in np.linspace(0, 1, granularity):
        if method == "linear":
            interp = (1 - mixture) * source + mixture * target
        elif method == "slerp":
            theta = tf.acos(tf.reduce_sum(source * target) /
                            (tf.norm(source) * tf.norm(target)))
            interp = (tf.sin((1 - mixture) * theta) / tf.sin(theta) * source +
                      tf.sin(mixture * theta) / tf.sin(theta) * target)
        else:
            raise ValueError("Invalid interpolation method specified: "
                             "{}".format(method))
        if gen:
            interp = gen(interp[tf.newaxis, :])
        imshow(interp)


def imshow(img, figsize=(12, 12)):
    """Wrapper for imshow.

    Parameters:
        img: Image to plot. Should be values between 0 and 1.
        figsize: Size of the figure window.

    """
    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(np.squeeze(img), cmap="Greys_r", vmin=0, vmax=1)
    plt.show()


def img_grid_npy(imgs, rows, cols, border_val=None, normalize=True):
    """
    Pack a bunch of image-like arrays into a grid for nicer visualization.

    Parameters:
        imgs: list of arrays, each height x width, or equivalent 3D array.
        rows: How many rows the grid should have.
        cols: Duh.
        border_val: Value to use as border between grid elements. If not given,
                    the maximum value over all images will be used.
        normalize: If true, every image will be individually normalized to a
                   maximum absolute value of 1.

    Returns:
         2d array you can use for matplotlib.

    """
    if len(imgs) != rows * cols:
        raise ValueError("Grid doesn't match the number of images!")

    if normalize:
        def norm(img):
            return img / np.abs(img).max()

        imgs = [norm(img) for img in imgs]
    imgs = np.asarray(imgs)
    if imgs.ndim == 4:
        multi_channel = [imgs.shape[-1]]
    else:
        multi_channel = []

    if border_val is None:
        border_val = imgs.max()

    # make border things
    col_border = np.full([imgs[0].shape[0], 1] + multi_channel,
                         border_val)

    # first create the rows
    def make_row(ind):
        base = imgs[ind:(ind + cols)]
        rborders = [col_border] * len(base)
        rinterleaved = [elem for pair in zip(base, rborders) for
                        elem in pair][:-1]  # remove last border
        return rinterleaved

    grid_rows = [np.concatenate(make_row(ind), axis=1) for
                 ind in range(0, len(imgs), cols)]

    # then stack them
    row_border = np.full([1, grid_rows[0].shape[1]] + multi_channel, border_val)
    borders = [row_border] * len(grid_rows)
    interleaved = [elem for pair in zip(grid_rows, borders) for
                   elem in pair][:-1]
    grid = np.concatenate(interleaved)
    return grid
