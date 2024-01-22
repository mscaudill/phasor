"""A collection of tools for working with numpy ndarrays."""

import numpy as np
import numpy.typing as npt

def normalize_axis(axis: int, ndim: int):
    """Returns a positive axis index for a supplied axis index of an ndim
    array.

    Args:
        axis:
            An positive or negative integer axis index.
        ndim:
            The number of dimensions to normalize axis by.
    """

    axes = np.arange(ndim)
    return axes[axis]


def islices(x: npt.NDArray, axis: int = -1):
    """An iterator yielding successive 1D slices from an ndarray.

    Args:
        x:
            An ndarray from which 1D slices will be made.
        axis:
            The axis of x containing the elements of each 1D slice.

    Examples:
        >>> x = np.arange(60).reshape(2, 3, 10)
        >>> print(x)
        >>> # elements along last axis will be elements in each slice
        >>> items = islice(x, axis=-1)
        >>> next(items)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> next(items)
        array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        >>> next(items)
        array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

    Yields:
        Successive 1-D slices whose elements match x's elements along axis for
        every 1D slice. The ordering of yieled arrays matches 'C' order in which
        the first axis changes slowest.
    """

    ax = normalize_axis(axis, x.ndim)

    idxs = [range(s) for axis, s in enumerate(x.shape) if axis != ax]
    for multiindex in itertools.product(*idxs):

        slicer = [slice(idx, idx+1) for idx in multiindex]
        slicer.insert(ax, slice(None))

        yield np.squeeze(x[tuple(slicer)])

