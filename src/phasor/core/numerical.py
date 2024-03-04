"""A collection of functions used to compute numerical results in Phasor."""

from typing import List, Union

import numpy as np
import numpy.typing as npt


def shannon(
    pdf: npt.NDArray,
    axis=-1,
    base=np.log10,
) -> Union[float, npt.NDArray]:
    """Returns the Shannon entropy of a discrete probability distribution.

    Args:
        pdf:
            An ndarray whose values along axis of all 1-D slices sum to 1 and
            are greater than or equal to 0.
        axis:
            The axis along which the entropy will be measured.
        base:
            The base units of the log for details see:
            https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Returns:
        An ndarray of 1 less dimension than pdf.
    """

    p = np.array(pdf)
    p[p==0] = 1
    return -1 * np.sum(p * base(p))




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

