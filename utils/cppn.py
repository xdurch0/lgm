from itertools import product

import numpy as np
import tensorflow as tf


def make_grid(res, dim, start=0, end=None):
    """Create a coordinate grid for CPPN input.

    Parameters:
        res: How many points in each dimension.
        dim: How many dimensions the grid should have.
        start: Leftmost point of the grid.
        end: Rightmost point of the grid. If not given, defaults to res-1.

    Returns:
        List with coordinate tuples.

    """
    seq = np.linspace(start, end if end else res-1, num=res, dtype=np.float32)
    seqs_nd = [seq]*dim
    return list(product(*seqs_nd))


def format_batch(batch, grid):
    """Prepare a batch of inputs into the proper format for a CPPN."""
    grid_tile = tf.tile(grid, [tf.shape(batch)[0], 1])
    batch_rep = repeat(batch, tf.shape(grid)[0])
    return tf.concat([grid_tile, batch_rep], axis=1)


def repeat(inp, times):
    """np.repeat equivalent.

    Currently only works for 2D inputs and repeats along axis 0!!

    TODO make less limited lol.
    """
    with tf.name_scope("repeat"):
        inp = inp[:, tf.newaxis, :]
        inp = tf.tile(inp, [1, times, 1])
        return tf.reshape(inp, [-1, tf.shape(inp)[2]])
