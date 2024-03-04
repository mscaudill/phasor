"""A module of tools for computing the phase and amplitude features from
time-frequency distributions or analytic signal transforms.

"""

from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.signal as sps

from openseize.filtering.bases import FIR
from openseize.filtering import fir


class Analytic:
    """A factory for producing phases & amplitudes from filtered analytic
    signals.

    An analytic signal is a complex valued signal with no negative frequency
    components encoding both phase & amplitude information. The Hilbert
    transform is used to compute analytic signals from real-valued (observed)
    signals. This tool comes with two important caveats. First, an analytic
    signal can be obtained on any real-valued signal but the phase is only
    meaningfully interpreted for narrow band (nearly sinusoidal) signals.
    Second, if you are looking for phase modulation of amplitude, the bandwidth
    of the filter used to measure the amplitude envelope must be at least twice
    the width of the center frequency at which the phase is measured (see Ref.
    4).

    Attributes:
        data:
            An n-dimensional array of real-valued signals.
        fs:
            The sampling rate at which data was measured.
        axis:
            The sample axis of data.
        ftype:
            An openseize FIR filter type for constructing bandpass filters for
            phase and amplitude extraction. Note, this is a FIR type not a FIR
            filter instance. Please see openseize.filtering.fir for all FIR
            filter options. Analytic does not support Remez exchange algorithm
            FIRs as they necessitate visually checks to verify filter
            performance.

    References:
        1. https://en.wikipedia.org/wiki/Analytic_signal
        2. https://en.wikipedia.org/wiki/Hilbert_transform
        3. Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
           phase-amplitude coupling between neuronal oscillations of different
           frequencies. J Neurophysiol. 2010 Aug; 104(2):1195-210. doi:
           10.1152/jn.00106.2010.
        4. Dvorak D, Fenton AA. Toward a proper estimation of phase-amplitude
           coupling in neural oscillations. J Neurosci Methods. 2014 Mar
           30;225:42-56. doi: 10.1016/j.jneumeth.2014.01.002.
    """

    def __init__(
        self,
        data: npt.NDArray,
        fs,
        axis: int = -1,
        ftype: FIR = fir.Kaiser):
        """Intialize this Analytic factory."""

        self.data = data
        self.fs = fs
        self.axis = axis
        self.nsamples = self.data.shape[self.axis]

        # we don't support Remez; its filter design needs to be visually chkd
        if ftype == fir.Remez:
            msg = 'The Remez filter is not a supported FIR filter type for {}'
            raise ValueError(msg.format(type(self))
        self.ftype = ftype

    def signal(
        self,
        fpass: Iterable[float],
        fstop: Iterable[float],
        **kwargs,
    ) -> npt.NDArray[np.complex_]
        """Returns the analytic signal over the passband frequencies

        Args:
            fpass:
                A 2-el iterable for the start & stop of the passband over
                which the analytic signal will be computed.
            fstop:
                A 2-el iterable for the start and stop of the stopband over
                which the analytic signal will be computed.
            **kwargs:
                Any valid kwarg for this instances filter type.

        Returns:
            An ndarray of the same shape as this instances data of the
            complex-valued analytic signal
        """

        filt = self.ftype(fpass, fstop, self.fs, **kwargs)
        filtered = filt(self.data, chunksize=self.nsamples, axis=self.axis)
        # compute the filtered analytic signal
        return sps.hilbert(filtered, N=None, axis=self.axis)

    def phases(
        self,
        fpass: Iterable[float],
        fstop: Iterable[float] = None,
        **kwargs,
    ) -> npt.NDArray:
        """Returns the phases of the analytic signal over the passband
        frequencies.

        The phase of an analytic signal is only meaningful over a narrowband. If
        fstop is provided it should have a narrow width about fpass.

        Args:
            fpass:
                A 2-el iterable for the start & stop of the passband over
                which the phases will be computed.
            fstop:
                A 2-el iterable for the start and stop of the stopband. If None,
                fstop will extend 2 Hz away from fpass (i.e. fstop = fpass
                + [-2, 2]).
            **kwargs:
                Any valid kwarg for this instances filter type.

        Returns:
            A real-valued array of phases of the same shape as this instances
            data attribute.
        """

        fpass = np.array(fpass)
        if fstop is None:
            fstop = fpass + np.array([-2, 2])

        # compute the filtered and nalytic signal
        analytic = self.signal(fpass, fstop, **kwargs)
        result = np.mod(np.angle(analytic) * 180 / np.pi, 360)

        return result

    def amplitudes(
        self,
        fpass: Iterable[float],
        fstop: Iterable[float],
        **kwargs,
    ) -> npt.NDArray:
        """Returns the amplitude envelopes of the analytic signal over the
        passband frequencies.

        The Fourier spectrum of an amplitude modulated signal will have
        components at fpass + [-fm, fm] where fm is the modulating frequency.
        To capture the full amplitude variation, the passband should include
        these modulating side frequencies. E.g. if the amplitude at 80 Hz is
        modulated by the phase at 10 Hz the pass band should run from 70 to 90
        Hz.

        Args:
           fpass:
                A 2-el iterable for the start & stop of the passband over
                which the amplitudes will be computed.
            fstop:
                A 2-el iterable for the start and stop of the stopband. If None,
                fstop will extend 2 Hz away from fpass (i.e. fstop = fpass
                + [-2, 2]).
            **kwargs:
                Any valid kwarg for this instances filter type.

        Returns:
            A real-valued array of amplitudes of the same shape as this instances
            data attribute.
        """

        fpass = np.array(fpass)
        fstop = np.array(fstop)

        # compute the filtered and nalytic signal
        analytic = self._analytic(fpass, fstop, **kwargs)

        return np.abs(analytic)


if __name__ == '__main__':

    from phasor.data.synthetic import PAC
    from openseize.filtering.fir import Kaiser
    from matplotlib import pyplot as plt

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=0.8)
    time, signal = pac(3, fs=500, shift=0, sigma=0.1, seed=0)

    hilbert = Hilbert(signal, fs=500)
    amplitudes = hilbert.amplitude(fpass=[40, 60], fstop=[35, 65], gpass=0.1,
            gstop=40) 
    phases = hilbert.phase(fpass=[8, 12], gpass=0.1, gstop=40)

    phase_kaiser = Kaiser(
        fpass=[8, 12], fstop=[6, 14], fs=500, gpass=0.1, gstop=40
    )
    theta_signal = phase_kaiser(signal, chunksize=len(signal), axis=-1)

    amp_kaiser = Kaiser(
        fpass=[40, 60], fstop=[35, 65], fs=500, gpass=0.1, gstop=40
    )
    gamma_signal = amp_kaiser(signal, chunksize=len(signal), axis=-1)


    fig, axarr = plt.subplots(5, 1, figsize=(6, 8))
    axarr[0].plot(time, signal, label='signal')
    axarr[1].plot(time, theta_signal, label='theta filtered')
    axarr[2].plot(time, phases, label='theta phases')
    axarr[3].plot(time, gamma_signal, label='Gamma filtered')
    axarr[3].plot(time, amplitudes, label='Gamma Amplitude')

    axarr[0].sharex(axarr[1])
    axarr[1].sharex(axarr[2])
    axarr[2].sharex(axarr[3])
    # show the magnitude of the Fourier spectrum
    freqs = np.fft.rfftfreq(len(signal), d=1 / 500)
    signal_fft = np.fft.rfft(signal)
    axarr[4].plot(freqs, np.abs(signal_fft), label='Spectrum')
    [ax.legend() for ax in axarr]
    plt.show()

