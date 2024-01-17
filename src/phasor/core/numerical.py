"""
"""

import itertools

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

def analytic(data, axis=-1, N=None):
    """Returns the Hilbert transform of data along axis.

    This function simply calls scipy's Hilbert. It's here because Scipy's
    naming convention is just wrong. Mathematically, a Hilbert transform
    computes only the imaginary part of an imaginary signal not the full
    analytic signal that scipy.signal.hilbert returns.

    Args:
        data:
            A numpy ndarray whose analytic signal will be returned.
        axis:
            The time or sample axis of data along which the Hilbert transform
            will be computed.
        N:
            The number of points used to estimate the FFT. If less than
            len(data[axis]) the data will be truncated. If greater than
            len(data[axis]) the data will be zero-padded. If None, the number of
            Fourier components will equal the length of data along axis.

    Notes:
        This function requires an in-memory data array. If your data is large
        consider openseize.filtering.special's hilbert function which
        approximates the Hilbert transform with a FIR filter that can return
        a producer of analytic signals.

    Returns:
        A complex array, the analytic signal of each 1-D array along axis.
    """

    return sps.hilbert(data, N=N, axis=axis)


def envelopes(analytic):
    """Returns the magnitude of the envelopes of an analytic signal.

    Args:
        analytic:
            A complex numpy array.

    Returns:
        The element-wise magnitude of analytic.
    """

    return np.abs(analytic)


def phases(analytic, deg=True):
    """Returns the phases of the analytic signal on the interval [0, 2*pi) or
    [0, 360).

    Args:
        analytic:
            A complex numpy array over which phases will be element-wise
            computed.
        deg:
            Boolean specifying if interval should be in radians [0, 2*pi) or
            degrees [0, 360). Default is degrees.

    Returns:
        An ndarray of phases with the same shape as analytic
    """

    # map phases from range (-pi, pi] to [0, 2*pi)
    x = np.mod(np.angle(analytic), 2*np.pi)
    if deg:
        x *= (180 / np.pi)
    return x


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




if __name__ == '__main__':

    import time

    """
    #x = np.random.random((40, 3, 500 * 90))
    amplitude = 2
    freq = 8
    fs = 500
    duration = 1
    phi = 0
    times = np.arange(0, fs *(duration + 1/fs)) / fs
    a = amplitude * np.sin(2 * np.pi * freq * times + phi)
    b = amplitude * np.sin(2 * np.pi * freq * times + phi+np.pi/2)
    x = np.stack((a, b))

    t0 = time.perf_counter()
    sa = analytic(x, axis=-1)
    print(f'Elapsed = {time.perf_counter() - t0} s')

    p = phases(sa)
    """

    x = np.random.randint(0, 17, size=(4,50))
    b = list(iter_slices(x))

