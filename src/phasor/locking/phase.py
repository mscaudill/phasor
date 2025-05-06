"""A module for computing various phase locking features from 1-D signals."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from openseize.filtering.bases import FIR
from openseize.filtering import fir

from phasor.transforms import analytic
from phasor.core.arraytools import standardize


class PhasePower:
    """A metric of the phase to power locking between two band-limited signals."""

    def __init__(
        self,
        data: npt.NDArray[np.float_],
        fs: float,
        axis: int = -1,
        ftype: FIR = fir.Kaiser,
        transform = analytic.CompactAnalytic
    ) -> None:
        """ """

        self.data = data
        self.fs = fs
        self.axis = axis
        self.ftype = ftype
        self.analytic = transform(data, fs, axis, ftype)


    def phases(self, fpass: Sequence[float], fstop: Sequence[float], **kwargs):
        """Returns phase signals for all channels in data."""

        return self.analytic.phase(fpass, fstop, **kwargs)


    def events(self, phase, angle=0, epsi=0.05):
        """Returns the indices of at which a single channels phase is near
        angle."""

        bools = np.logical_and(phase > angle - epsi, phase < angle + epsi)

        return np.flatnonzero(bools)

    def powers(
        self,
        fstart=40,
        fstop=250,
        fstep=2,
        bandwidth=4,
        ):
        """ """


        result = []
        for center in range(fstart, fstop, fstep):
            print(center)
            fpass = center + np.array([-bandwidth, bandwidth])
            fstop = fpass + np.array([-bandwidth, bandwidth])
            amplitudes = self.analytic.envelope(fpass, fstop)
            #amplitudes = standardize(amplitudes, axis=self.axis)
            result.append(amplitudes ** 2)

        return np.stack(result).swapaxes(0,1)


    # here phases and powers are now assumed to be for a single channel
    def comodulation(self, phase, power, angle=0, epsi=0.01, binsize=1000):
        """ """

        locs = self.events(phase, angle, epsi)
        locs = locs[np.logical_and(locs > binsize // 2, locs < len(phase)
            - binsize//2)]

        segments = [power[:, loc-binsize//2 : loc+binsize//2] for loc in locs]
        return np.sum(segments, axis=0) / len(segments)
        #return segments





if __name__ == '__main__':

    import time
    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    from openseize.filtering.iir import Notch


    import matplotlib.pyplot as plt

    fp = ('/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    'No_6489_left_2022-02-09_13_55_22_(2)_annotations.edf')


    # left hemisphere chs 1, 2, 3
    # right hemisphere chs 0, 2, 3

    with Reader(fp) as reader:
        data = reader.read(start=0)

    # downsample to 500 Hz
    x = downsample(data, M=10, fs=5000, chunksize=data.shape[-1])
    # notch filter data
    notch = Notch(fstop=60, width=4, fs=500)
    x = notch(x, chunksize=x.shape[-1])

    phasepower = PhasePower(x, fs=500)

    phases = phasepower.phases(fpass=[4, 12], fstop=[2, 14])
    powers = phasepower.powers(fstop=160)


    result = phasepower.comodulation(phases[1], powers[1], angle=0)

    fig, ax = plt.subplots()
    time = np.linspace(-500, 500, 1000)
    freqs = np.arange(40, 160, 2)
    mesh = ax.pcolormesh(time, freqs, result)
    plt.show()
