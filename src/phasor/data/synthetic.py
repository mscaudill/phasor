""" """

import numpy as np


class PACSignal:
    """A callable consisting of two coupled oscillations with the phase
    of one oscillation modulating the amplitude of the other oscillation.

    The call method allows this signal to be constructed for any duration and
    sampling rate.

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
        sigma:
            The standard deviation of additive noise to this signal.
        seed:
            A random seed integer for generating random but reproducible
            signals.
    """

    def __init__(self, fp, fa, amp_p, amp_a, strength, sigma, seed=None):
        """Initialize this signal."""

        self.fp = fp
        self.fa = fa
        self.amp_p = amp_p
        self.amp_a = amp_a
        self.strength = strength
        self.sigma = sigma
        self.seed = seed

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

    def noise(self, time):
        """Returns the Gaussian white noise component of this signal.

        Args:
            time:
                A 1-D vector of times over which this component will be
                computed.

        Returns:
            A 1-D array of the same length as time.
        """

        rng = np.random.default_rng(self.seed)
        return rng.normal(scale=self.sigma, size=len(time))

    def __call__(self, duration, fs):
        """Returns a 1-D array of times and a 1-D array of PAC signal values.

        Args:
            duration:
                The duration of the signal to create in seconds.
            fs:
                The sampling rate of the signal to create in Hz.

        Returns:
            A 2-tuple of 1-D arrays, the times array and the PACSignal array.
        """

        time = np.arange(0, duration + 1 / fs, 1 / fs)
        return time, self.modulated(time) + self.phasic(time) + self.noise(time)







if __name__ == '__main__':


    # FIXME I want to take hilbert and show the density plot

    pac = PACSignal(fp=8, fa=40, amp_p=0.5, amp_a=1, strength=0.6, sigma=0.3,
            seed=0)
    time, signal = pac(1, fs=500)

    import matplotlib.pyplot as plt
    plt.plot(time, signal)
    plt.show()
