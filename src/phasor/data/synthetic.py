"""

"""

from abc import abstractmethod
from collections import abc
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt


class Signal(abc.Callable):
    """An ABC defining concrete and required methods of all synthetic Signals.

    As an ABC, this class can not be instantiated. To create a specific signal
    instance use one of the inheriting signal classes.
    """

    def noise(
            self,
            size: Union[int, Tuple],
            sigma: float,
            seed: Optional[int] = None,
    ) -> npt.NDArray:
        """Returns an array of normally distributed values with standard
        deviation sigma.

        Args:
            size:
                The output shape of the noise signal.
            sigma:
                The standard deviation of the normally distributed values to
                return.
            seed:
                An integer seed for numpy's random number generator for random
                but reproducible arrays. If None, values are irreproducible.

        Returns:
            A 'size' shape ndarray of random normal values.

        Notes:
            see numpy.random.generator.normal for further details.
        """

        rng = np.random.default_rng(seed)
        return rng.normal(scale=sigma, size=size)

    @abstractmethod
    def __call__(
            self,
            duration: float,
            fs: float,
            sigma: Optional[float] = None,
            seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns an array of signal values of length duration * fs.

        Args:
            duration:
                The duration in seconds of the created signal.
            fs:
                The sampling rate in Hz of the created signal.

        Returns:
            A 2-tuple of times and a numpy array of signal values.
        """


class MultiSine(Signal):
    """A callable for creating a signal containing multiple sine waves.

    The call method allows this signal to be constructed for any duration,
    sampling rate and noise level.

    Attributes:
        amps:
            A sequence or 1-D array of amplitudes one per sine in this Signal.
        freqs:
            A sequence or 1-D array of frequencies, one per sine in this Signal.
        times:
            A 2-D array of start and stop times, one per sine in this Signal.
            The stop time is exclusive (i.e. an open interval [start, stop)).
    """

    def __init__(
            self,
            amps: Union[Sequence, npt.NDArray],
            freqs: Union[Sequence, npt.NDArray],
            times: Union[Sequence, npt.NDArray],
    ) -> None:
        """Initialize this MultiSine Signal."""

        self.amps = np.array(amps)
        self.freqs = np.array(freqs)
        self.times = np.atleast_2d(times)

    def __call__(
            self,
            duration: float,
            fs: float,
            sigma: Optional[float] = None,
            seed: Optional[int] = None,
    ) -> npt.NDArray:
        """Returns a 1-D array of length duration in seconds containing multiple
        sine waves.

        Args:
            duration:
                The duration of the signal to create in seconds.
            fs:
                The sampling rate of the signal in Hz.
            sigma:
                The standard deviation of normally distributed noise to add to
                signal. If None, no noise is added.
            seed:
                A random seed integer for creating reproducible but random
                signals. If None, signal will be random but irreproducible.

        Returns:
            A 2-tuple of times and a numpy array of signal values.
        """

        signal = np.zeros(duration * fs + 1)
        times = np.linspace(0, duration, duration * fs + 1)
        for amp, freq, (a, b) in zip(self.amps, self.freqs, self.times):
            signal[a * fs: b * fs] += np.sin(
                    2 * np.pi * freq * np.arange(b-a, step=1/fs)
                    )

        noise = self.noise(duration * fs + 1, sigma, seed) if sigma else 0
        signal += noise

        return times, signal


class LinearChirp(Signal):
    """A callable for constructing a linear chirp Signal.

    A linear chirp is a sine wave whose frequency linearly increases between
    a start and stop time.

    The call method allows this signal to be constructed for any duration,
    sampling rate and noise level.

    Attributes:
        amp:
            The amplitude of the created chirp.
        start:
            The start time of this chirp.
        stop:
            The stop time of this chirp.
        f_range:
            A 2-el sequence of start and stop frequencies in this chirp.

    References:
        1. https://en.wikipedia.org/wiki/Chirp
    """

    def __init__(
            self,
            amp: float,
            start: float,
            stop: float,
            f_range: Tuple[float, float],
    ) -> None:
        """Initialize this LinearChirp Signal."""

        self.amp = amp
        self.start = start
        self.stop = stop
        self.f_range = f_range

    def __call__(
            self,
            duration: float,
            fs: float,
            sigma: Optional[float] = None,
            seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns a 1-D array of length duration seconds containing a chirp
        between start and stop.

        Args:
            duration:
                The duration of the signal to create in seconds.
            fs:
                The sampling rate of the signal in Hz.
            sigma:
                The standard deviation of normally distributed noise to add to
                signal. If None, no noise is added.
            seed:
                A random seed integer for creating reproducible but random
                signals. If None, signal will be random but irreproducible.

        Returns:
            A 2-tuple of times and a numpy array of signal values.
        """

        signal = np.zeros(duration * fs + 1)
        times = np.linspace(0, duration, duration * fs + 1)

        # get the chirp rate and compute the phase (integral of freq change)
        rate = np.diff(self.f_range) / (self.stop - self.start)
        t = np.arange(self.stop - self.start, step=1/fs)
        phase = 2 * np.pi * (rate/2 * t**2 + self.f_range[0] * t)
        signal[self.start * fs: self.stop * fs] = np.sin(phase)

        noise = self.noise(duration * fs + 1, sigma, seed) if sigma else 0
        signal += noise

        return times, signal


# FIXME would be nice to specify a start and stop here like other signals
# FIXME add the Tort reference
class PAC(Signal):
    """A callable consisting of two coupled oscillations with the phase
    of one oscillation modulating the amplitude of the other oscillation.

    The call method allows this signal to be constructed for any duration,
    sampling rate and noise level.

    Attributes:
        fp:
            The frequency, in Hz, of the oscillation whose phase modulates the
            amplitude of the other oscillation.
        fa:
            The frequency, in Hz, of the oscillation whose amplitude is
            modulated by the phase of the other oscillation.
        amp_p:
            The max amplitude of the oscillation whose phase modulates the
            amplitude of the other oscillation.
        amp_a:
            The max amplitude of the oscillation whose amplitude is
            modulated by the phase of the other oscillation.
        strength:
            A value in [0,1] that encodes how strongly the amplitude of one
            oscillation is driven by the phase of the other oscillation. If 0,
            the amplitude is not driven by phase at all. If 1, the amplitude is
            completely driven by the phase.
    """

    def __init__(self, fp, fa, amp_p, amp_a, strength):
        """Initialize this signal."""

        self.fp = fp
        self.fa = fa
        self.amp_p = amp_p
        self.amp_a = amp_a
        self.strength = strength

    def modulated(self, time):
        """Returns the amplitude modulated component of the signal over a vector
        of times.

        Args:
            time:
                A 1-D vector of times over which this component will be
                computed.

        Returns:
            A 1-D array of the same length as time.
        """

        chi = 1 - self.strength
        mod = ((1- chi) * np.sin(2 * np.pi * self.fp * time) + 1 + chi) / 2
        return self.amp_a * mod * np.sin(2 * np.pi * self.fp * time)

    def phasic(self, time):
        """Returns the phase modulating component of this signal.

        Args:
            time:
                A 1-D vector of times over which this component will be
                computed.

        Returns:
            A 1-D array of the same length as time.
        """

        return self.amp_p * np.sin(2 * np.pi * self.fp * time)

    def __call__(self, duration, fs, sigma=None, seed=None):
        """Returns a 1-D array of times and a 1-D array of PAC signal values.

        Args:
            duration:
                The duration of the signal to create in seconds.
            fs:
                The sampling rate of the signal to create in Hz.
            sigma:
                The standard deviation of additive noise to this signal.
            seed:
                A random seed integer for generating random but reproducible
                signals.

        Returns:
            A 2-tuple of 1-D arrays, the times array and the PACSignal array.
        """

        time = np.linspace(0, duration, duration * fs + 1)
        noise = self.noise(duration * fs + 1, sigma, seed) if sigma else 0
        return time, self.modulated(time) + self.phasic(time) + noise







if __name__ == '__main__':


    """
    pac = PAC(fp=8, fa=40, amp_p=0.5, amp_a=1, strength=0.6)
    time, signal = pac(1, fs=500, sigma=0.1)
    """

    """
    msine = MultiSine(amps=[1,1,1], freqs=[4.5, 9, 12], times=[[6, 10], [6, 10],
        [2,5]])
    time, signal = msine(duration=12, fs=128, sigma=0.01, seed=None)
    """

    chirp = LinearChirp(amp=1, start=2, stop=6, f_range=(2,20))
    time, signal = chirp(duration=12, fs=128, sigma=.05, seed=0)

    import matplotlib.pyplot as plt
    plt.plot(time, signal)
    plt.show()
