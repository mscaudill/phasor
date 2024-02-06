"""A collection of tools for working with numpy ndarrays."""

from typing import Sequence

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

def pad_along_axis(
    arr: npt.NDArray, pad: Sequence, axis: int = -1, **kwargs
) -> npt.NDArray:
    """Wrapper for numpy pad allowing before and after padding along
    a single axis.

    Args:
        arr (ndarray):              ndarray to pad
        pad (int or array-like):    number of pads to apply before the 0th
                                    and after the last index of array along
                                    axis. If int, pad number of pads will be
                                    added to both
        axis (int):                 axis of arr along which to apply pad.
                                    Default pads along last axis.
        **kwargs:                   any valid kwarg for np.pad
    """

    # convert int pad to seq. of pads & place along axis of pads
    pad = [pad, pad] if isinstance(pad, int) else pad
    pads = [(0, 0)] * arr.ndim
    pads[axis] = pad
    return np.pad(arr, pads, **kwargs)

def pad_axis_to(
    arr: npt.NDArray,
    amt: int,
    side: str,
    axis: int = -1,
    **kwargs,
) -> npt.NDArray:
    """Wrapper for numpy pad that allows for padding upto an integer length
    along an axis.

    Args:
        arr:
            A ndarray to pad along axis.
        amt:
            The desired length of arr along axis after padding.
        side:
            The side of array along axis which to pad. Must be one of 'left',
            'right' or 'center'. 'left' pads the left side of axis of arr to
            achieve amt length result, 'right' pads the right side of arr to
            achieve amt length result and center pads the left and right by
            equal amounts to achieve amt length result.
        axis:
            The axis along which the padding will be carried out.
        kwargs:
            Any valid kwarg for numpy's pad function.

    Examples:
        >>> x = np.ones((3, 4), dtype=int)
        >>> pad_axis_to(x, 2, side='right')
        array([[1, 1, 1, 1, 0, 0],
               [1, 1, 1, 1, 0, 0],
               [1, 1, 1, 1, 0, 0]])
        >>> pad_axis_to(x, 1, side='left')
        array([[0, 1, 1, 1, 1],
               [0, 1, 1, 1, 1],
               [0, 1, 1, 1, 1]])
        >>> # uneven center padding places unpaired pad on right
        >>> pad_along_axis(x, 3, side='center')
        array([[0, 1, 1, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 0, 0]])

    Returns:
        An ndarray whose shape matches arr on all axes except axis is length is
        extended to amt.

    Raises:
        A ValueError is issued if side is not in {'left', 'right', 'center'}.
    """

    sides = {'left', 'right', 'center'}
    if side.lower() not in sides:
        raise ValueError(f'side must be one of {sides}')

    pad_amt = amt - arr.shape[axis]
    pads = {
        'left': [pad_amt, 0],
        'right': [0, pad_amt],
        'center': [pad_amt // 2, pad_amt // 2 + pad_amt % 2],
    }
    return pad_along_axis(arr, pads[side.lower()], axis=axis, **kwargs)

