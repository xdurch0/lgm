import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def random_samples(model, noise_fn, img_dims=(32, 32), num=10):
    """Generate and plot some random samples.

    Parameters:
        model: Callable that produces images, e.g. Keras Sequential.
        noise_fn: Callable that produces a noise batch, should take (only) batch
                  size as argument.
        img_dims: Target dimensions of each image.
        num: How many samples to plot.

    """
    samples = model(noise_fn(num))
    for sample in samples:
        imshow(sample, img_dims)


def random_sample_grid(model, noise_fn, img_dims=(32, 32), grid_dims=(4, 4),
                       show=True):
    """Construct a grid of some random samples in a single image.

    Parameters:
        model: Callable that produces images, e.g. Keras Sequential.
        noise_fn: Callable that produces a noise batch, should take (only) batch
                  size as argument.
        img_dims: Target dimensions of each image.
        grid_dims: Desired grid dimensions, also specifying how many samples to
              generate.
        show: If true, plot the grid.

    Returns:
        Numpy array with the grid.

    """
    if len(grid_dims) != 2:
        raise ValueError("Grid dimension needs to be 2D.")
    to_gen = np.prod(grid_dims)
    samples = model(noise_fn(to_gen))
    samples = [sample.numpy().reshape(img_dims) for sample in samples]

    grid = img_grid_npy(samples, grid_dims[0], grid_dims[1], normalize=False)
    if show:
        imshow(grid)
    return grid


def interpolate(source, target, img_dims=(32, 32), granularity=20, gen=None,
                method="linear"):
    """Interpolate between two images, showing intermediate results.

    Parameters:
        source: Starting point.
        target: End point.
        img_dims: If given, reshape...
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
        imshow(interp, img_dims)


def imshow(img, target_dims=None):
    """Wrapper for imshow.

    Parameters:
        img: Image to plot. Should be values between 0 and 1.
        target_dims: If None, image is assumed to have proper dimensions. If
                     given, should be a tuple that the image will be reshaped
                     to.

    """
    if target_dims is not None:
        img = img.numpy().reshape(target_dims)
    plt.imshow(img, cmap="Greys_r", vmin=0, vmax=1)
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

    if border_val is None:
        border_val = imgs.max()

    # make border things
    col_border = np.full([imgs[0].shape[0], 1], border_val)

    # first create the rows
    def make_row(ind):
        base = imgs[ind:(ind + cols)]
        borders = [col_border] * len(base)
        interleaved = [elem for pair in zip(base, borders) for
                       elem in pair][:-1]  # remove last border
        return interleaved

    grid_rows = [np.concatenate(make_row(ind), axis=1) for
                 ind in range(0, len(imgs), cols)]

    # then stack them
    row_border = np.full([1, grid_rows[0].shape[1]], border_val)
    borders = [row_border] * len(grid_rows)
    interleaved = [elem for pair in zip(grid_rows, borders) for
                   elem in pair][:-1]
    grid = np.concatenate(interleaved)
    return grid
