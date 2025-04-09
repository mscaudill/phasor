"""An analytic signal transform for extracting the amplitude and phase
components of real signals."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from openseize.filtering.bases import FIR
from openseize.filtering import fir


class CompactAnalytic:
    """The band-limited Hilbert transform of a real-valued signal.

    Signals have analytic representations only when the Fourier transform has
    compact frequency support (i.e. band-limited). The amplitude and phase
    methods of this transform accept filter passbands to compute the analytic
    representation over band-limited frequency ranges.

    Attributes:
        data:
            A 2-D numpy array of 1-D signals
        fs:
            The sampling rate of the signals in data.
        axis:
            The sample (i.e. time) axis of data.
        ftype:
            An Openseize FIR filter for band-limiting the signals prior to
            applying the Hilbert transform. Defaults to a Kaiser.

    References:
        1. https://en.wikipedia.org/wiki/Analytic_signal
        2. https://en.wikipedia.org/wiki/Hilbert_transform
        3. Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
           phase-amplitude coupling between neuronal oscillations of different
           frequencies. J Neurophysiol. 2010 Aug; 104(2):1195-210. doi:
           10.1152/jn.00106.2010.
    """

    def __init__(
        self,
        data: npt.NDArray,
        fs: float,
        axis: int = -1,
        ftype: FIR = fir.Kaiser,
    ) -> None:
        """Initialize this transform."""

        self.data = data
        self.fs = fs
        self.axis = axis
        self.ftype = ftype


    def signal(
        self,
        fpass: Sequence[float],
        fstop: Sequence[float],
        **kwargs,
    ) -> npt.NDArray[np.complex_]:
        """Returns the analytic representation of each signal in data  in the passband frequencies.

        Args:
            fpass:
                A 2-el sequence of start and stop passband frequencies.
            fstop:
                A 2-el sequence of start and stop stopband frequences. The
                difference of these frequencies with fpass define the transition
                widths.
            **kwargs:
                Any valid kwargs for this transforms filter.

        Returns:
            A complex ndarray of the same shape as this transforms data
            attribute.
        """

        n = self.data.shape[self.axis]
        filt = self.ftype(fpass, fstop, self.fs, **kwargs)
        filtered = filt(self.data, chunksize=n, axis=self.axis)
        # FIXME
        # need to add optional normalization
        mu = np.mean(filtered, axis=self.axis, keepdims=True)
        std = np.std(filtered, axis=self.axis, keepdims=True)
        normed = (filtered - mu) / std

        return sps.hilbert(normed, axis=self.axis)


    def envelope(
        self,
        fpass: Sequence[float],
        fstop: Sequence[float],
        **kwargs,
    ) -> npt.NDArray[np.float_]:
        """Returns the envelope amplitudes of the real component of the analytic
        signal in the passband frequencies.

        Args:
            fpass:
                A 2-el sequence of start and stop passband frequencies.
            fstop:
                A 2-el sequence of start and stop stopband frequences. The
                difference of these frequencies with fpass define the transition
                widths.
            **kwargs:
                Any valid kwargs for this transforms filter.

        Returns:
            A real-valued array of envelope amplitudes the same shape as data.
        """

        # compute the filtered and analytic signal
        analytic = self.signal(fpass, fstop, **kwargs)

        return np.abs(analytic)


    def phase(
        self,
        fpass: Sequence[float],
        fstop: Sequence[float],
        **kwargs,
    ) -> npt.NDArray:
        """Returns the phases in [0, 2*pi) of the analytic signal over the passband
        frequencies.

        The phase of an analytic signal is only meaningful over a narrowband. If
        fstop is provided it should have a narrow width about fpass.

        Args:
            fpass:
                A 2-el sequence of start and stop passband frequencies.
            fstop:
                A 2-el sequence of start and stop stopband frequences. The
                difference of these frequencies with fpass define the transition
                widths.
            **kwargs:
                Any valid kwargs for this transforms filter.


        Returns:
            A real-valued array of phases in radians with the same shape as data.
        """

        # compute the filtered and nalytic signal
        analytic = self.signal(fpass, fstop, **kwargs)
        result = np.angle(analytic)
        # convert to [0, 2*pi)
        result[result < 0] += 2 * np.pi

        return result


if __name__ == '__main__':


    from phasor.data.synthetic import PAC
    from openseize.filtering.fir import Kaiser
    from matplotlib import pyplot as plt

    import scipy.signal as sps

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=0.8)
    time, signal = pac(3, fs=500, shift=0, sigma=0.1, seed=0)

    hilbert = CompactAnalytic(signal, fs=500)
    amplitudes = hilbert.envelope(fpass=[40, 60], fstop=[35, 65]) 
    phases = hilbert.phase(fpass=[4, 10], fstop=[2, 12])


    # compute filtered signals for plotting
    phase_kaiser = Kaiser(fpass=[4, 10], fstop=[2, 12], fs=500)
    theta_signal = phase_kaiser(signal, chunksize=len(signal), axis=-1)

    amp_kaiser = Kaiser(fpass=[40, 60], fstop=[35, 65], fs=500)
    gamma_signal = amp_kaiser(signal, chunksize=len(signal), axis=-1)


    fig, axarr = plt.subplots(4, 1, figsize=(6, 8))
    axarr[0].plot(time, signal, label='signal')
    axarr[1].plot(time, theta_signal, label='theta filtered')
    axarr[2].plot(time, phases, label='theta phases')
    axarr[3].plot(time, gamma_signal, label='Gamma filtered')
    axarr[3].plot(time, amplitudes, label='Gamma Amplitude')

    axarr[0].sharex(axarr[1])
    axarr[1].sharex(axarr[2])
    axarr[2].sharex(axarr[3])

