"""A module for computing various phase locking features from 1-D signals."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from openseize.filtering.bases import FIR
from openseize.filtering import fir

from phasor.transforms.analytic import Analytic
from phasor.core.arraytools import standardize


class PhaseToPower:
    """An estimator of phase-locked power in local field potential recording(s).

    This estimator estimates the amount of Phase-locking between the phase of
    a signal in a narrow freequency band to the amplitude of a signal in
    a different narrow frequency band using the Hilbert transform.
    """

    def __init__(
        self,
        data: npt.NDArray[np.float_],
        fs: float,
        axis: int = -1,
        ftype: FIR = fir.Kaiser,
    ) -> None:
        """Intialize this estimator.

        Args:
        """

        self.data = data
        self.fs = fs
        self.axis = axis
        self.ftype = ftype
        self.analytic = Analytic(data, fs, axis, ftype)


    def phase(self, fpass: Sequence[float], fstop: Sequence[float], **kwargs):
        """Returns the phase time-series for all channels in data."""

        return self.analytic.phase(fpass, fstop, **kwargs)


    def indices(self, phase, angle: float = 0, epsi: float = 0.05):
        """Returns sample indices of data where phase is epsi close to angle.

        The angle is measure in [0, 2 * pi] radians.

        Args:

        """

        minimum, maximum = angle + np.array([-epsi, epsi])

        return np.flatnonzero(phase > minimum, phase < maximum)


    def _amplitude(self, center, bandwidth, standardize, **kwargs):
        """ """

        fpass = center + np.array([-bandwidth, bandwidth])
        fstop = fpass + np.array([-bandwidth, bandwidth])

        return self.analytic.envelope(fpass, fstop, standardize, **kwargs)


    def amplitudes(






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



# TODO 5/6 start Formal class to refactor above



if __name__ == '__main__':

    import time
    from openseize.file_io.edf import Reader
    from openseize.resampling.resampling import downsample
    from openseize.filtering.iir import Notch


    import matplotlib.pyplot as plt

    base = '/media/matt/Magnus/Qi/EEG_annotation_03272024/'
    #base = '/home/matt/python/nri/data/rett_eeg/pretreated/edf/'


    # WT SHAM
    #name = 'No_6489_right_2022-02-09_14_58_21_(2)_annotations.edf'
    #name = 'No_6493_right_2022-02-08_11_06_48_annotations.edf'
    #name = 'No_6495_right_2022-02-09_14_58_20_(2)_annotations.edf'
    #name = 'No_6503_right_2022-02-08_15_27_48_annotations.edf'
    #name = 'No_6506_right_2022-02-09_10_29_36_annotations.edf'
    #name = 'No_6511_right_2022-02-09_12_48_26_annotations.edf'

    # RTT SHAM
    #name = 'No_6492_right_2022-02-08_11_06_46_annotations.edf'
    #name = 'No_6496_right_2022-02-08_13_20_14_annotations.edf'
    #name = 'No_6501_right_2022-02-08_15_27_52_annotations.edf'
    name = 'No_6502_right_2022-02-08_15_27_50_annotations.edf'
    #name = 'No_6505_right_2022-02-09_10_29_34_annotations.edf'
    #name = 'No_6508_right_2022-02-09_12_48_22_annotations.edf'

    # RTT DBS
    #name = 'No_6491_right_2022-02-07_17_05_20_annotations.edf'
    #name = 'No_6497_right_2022-02-08_13_20_15_annotations.edf'
    #name = 'No_6498_right_2022-02-08_13_20_16_annotations.edf'
    #name = 'No_6507_right_2022-02-09_10_29_37_annotations.edf'
    #name = 'No_6509_right_2022-02-09_12_48_24_annotations.edf'
    #name = 'No_6512_right_2022-02-09_14_58_19_annotations.edf'



    # OLD data set
    # WT-sham
    #name = '5881_Right_group B.edf'

    fp = base + name

    # left hemisphere chs 1, 2, 3 -> DMS1, DMS2, M1
    # right hemisphere chs 0, 2, 3 -> DMS1, DMS2, M2

    phase_pass = [4, 12]
    phase_stop = [2, 14]
    amp_start = 12
    amp_stop = 240
    amp_step = 2
    binsize = 1000
    channels = [3, 0]

    with Reader(fp) as reader:
        data = reader.read(start=0)

    # downsample to 500 Hz
    x = downsample(data, M=10, fs=5000, chunksize=data.shape[-1])

    # FIXME openseize needs a comb filter -- make one in phasor
    # notch filter data at 60, 120 and 180 Hz
    notch1 = Notch(fstop=60, width=4, fs=500)
    notch2 = Notch(fstop=120, width=4, fs=500)
    notch3 = Notch(fstop=180, width=4, fs=500)
    x = notch1(x, chunksize=x.shape[-1])
    x = notch2(x, chunksize=x.shape[-1])
    x = notch3(x, chunksize=x.shape[-1])

    phasepower = PhasePower(x, fs=500)

    phases = phasepower.phases(fpass=phase_pass, fstop=phase_stop)
    powers = phasepower.powers(fstart=amp_start, fstop=amp_stop, fstep=amp_step)

    # FIXME the phases and powers are computed for each channels so no need to
    # recompute just to plot. You should store phases and powers to PhasePower
    # instance and build a plot method that takes specific channels
    result = phasepower.comodulation(
                phases[channels[0]],
                powers[channels[1]],
                angle=0
                )

    fig, ax = plt.subplots()
    time = np.linspace(-binsize // 2, binsize // 2, binsize)
    freqs = np.arange(amp_start, amp_stop, amp_step)
    mesh = ax.pcolormesh(time, freqs, result, cmap='jet')
    ax.set_title(name + ' ' + str(channels))
    cb = plt.colorbar(mesh)
    plt.show()
