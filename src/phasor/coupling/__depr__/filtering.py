"""Module of tools for filtering data prior to the Hilbert transform.

Classes:
    Multifilter:
        A callable that constructs a sequence of bandpass filters that can be
        sequentially called on a numpy ndarray.
"""

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from openseize.filtering import fir
from openseize.filtering.bases import FIR

from phasor.core.mixins import ReprMixin


class Multifilter(ReprMixin):
    """A callable for constructing and executing a sequence of FIR bandpass
    filters on a numpy ndarray.

    Attrs:
        ftype:
            An openseize finite impulse response filter.
        fs:
            The sampling rate of the data to be filtered.
        centers:
            A sequence or ndarray of center frequencies for each FIR bandpass
            filter.
        width:
            The width in Hz between the -6dB (50% attenuation) filter cutoffs.
        transition:
            The width of each transition band.
        gpass:
            The maximum pass band ripple in dB. Defaults to 0.25 dB ~ 3% ripple.
        gstop:
            The minimum stop band attenuation in dB. Defaults to 40 ~99%
            attenuation.
        fpasses:
            A len(centers) x 2 array of pass bands whose edges are (center
            - width/2, center + width/2) for each center in centers.
        fstops:
            A len(centers) x 2 array of stop bands whose edges are (center
            - width/2 -transition, center + width/2 + transition) for each
              center in centers.

    Properties:
        filters:
            A sequence of FIR filters matching each pass and stop band
            specification.

    Examples:
        >>> # make a MultiFilter with 2 passbands with 4 Hz widths at -6dB
        >>> multifilter = Multifilter(fs=500, centers=[4, 5], width=2,
        ...                           transition=2)
        >>> multifilter.fpasses
        array([[3., 5.],
               [4., 6.]])
        >>> multifilter.fstops # transition band edges
        array([[1., 7.],
               [2., 8.]])
        >>> multifilter.filters[0].cutoff # -6 dB points (4Hz width)
        array([2., 6.])
        >>> # build a data array 40 trials x 3 channels x 90 secs @ 500Hz
        >>> data = np.random.random((40, 3, 90*500))
        >>> # execute both bandpass filters along sample axis
        >>> result = multifilter(data, axis=-1)
        >>> print(result.shape)
        (2, 40, 3, 45000)
    """

    def __init__(
        self,
        fs: int,
        centers: Union[Sequence, npt.NDArray],
        width: float,
        transition: float,
        ftype: FIR = fir.Kaiser,
        gpass: float = 0.25,
        gstop: float = 40,
    ) -> None:
        """Initialize this Multifilter instance."""

        self.ftype = ftype
        self.fs = fs
        self.centers = np.array(centers)
        self.width = width
        self.transition = transition
        self.gpass = gpass
        self.gstop = gstop

        self.fpasses = np.stack(
            [self.centers - width / 2, self.centers + width / 2], axis=1
        )
        self.fstops = self.fpasses + [-transition, transition]

    @property
    def filters(self):
        """Returns a list of FIR filters meeting this Multifilter's
        specifications."""

        result = []
        for fpass, fstop in zip(self.fpasses, self.fstops):
            filt = self.ftype(fpass, fstop, self.fs, self.gpass, self.gstop)
            result.append(filt)
        return result

    def __call__(self, data: npt.NDArray, axis: int = -1):
        """Convolves each of this Multifilter's FIR filters on data along axis.

        Args:
            data:
                An ndarray to be filtered by each filter in this Multifilter.
            axis:
                The sample axis along which convolution should occur. Defaults
                to the last axis.

        Returns:
            An ndarray of shape:
            (len(centers), data.shape[0], data.shape[1], ...)
        """

        result = []
        for filt in self.filters:
            result.append(filt(data, chunksize=data.shape[axis], axis=axis))

        return np.stack(result)


if __name__ == '__main__':
    import time

    mfilter = Multifilter(
        fs=500, centers=np.arange(4, 31), width=2, transition=2
    )

    x = np.random.random((40, 3, 500 * 90))
    t0 = time.perf_counter()
    res = mfilter(x, axis=-1)
    elapsed = time.perf_counter() - t0
    print(f'Computed {elapsed} s')
