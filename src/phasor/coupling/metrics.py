"""A collection of cross-frequency coupling metrics.

"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from phasor.core import numerical


class ModulationIndex:
    """A phase-amplitude coupling metric callable based on the distance between
    the observed phase-amplitude distribution and a uniform distribution.

    This is the method of Tort et. al. 2010. It constructs an observed
    phase-amplitude distribution by binning the phases and computing the average
    amplitude within each phase bin. The Kullback-Leibler divergence is then
    used to compare this distribution to a uniform distribution.

    Attributes:
        num_bins:
            The number of bins used to partition the phases which are assumed to
            be in degree units [0, 360].
        binsize:
            The size of each bin in degrees.
        phase_bins:
            A 1-D array of right-side bin edges in [binsize, 360].
        densities:
            A 1-D array of phase-amplitude densities or None. This attribute is
            initialized None and set to a 1-D array after this instance is called
            on a set of phases and amplitudes.

    References:
        1. Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
           phase-amplitude coupling between neuronal oscillations of different
           frequencies. J Neurophysiol. 2010 Aug; 104(2):1195-210. doi:
           10.1152/jn.00106.2010.
    """

    def __init__(self, num_bins: int):
        """Initialize metric with the number of bins to partition the phases.

        Args:
            num_bins:
                An integer number of bins to partition the phases into.

        Returns:
            None but stores the number of bins and an array of bins to this
            instance.
        """

        self.num_bins = num_bins
        self.binsize = 360 / self.num_bins
        # bins are defined by their largest (i.e. rightmost) values
        self.phase_bins = np.linspace(self.binsize, 360, self.num_bins)
        self.densities = None

    # ax is phasors standard naming convention for mpl Axes instances
    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[None, plt.Axes]:
        """Constructs a plot of this instances phase-amplitude distribution.

        Args:
            ax:
                A matplotlib Axes for plotting the binned phases & mean
                amplitudes to. If None, a new figure & axis is created.
            kwargs:
                Any valid kwarg for matplotlib's bar function.

        Returns:
            An Axes instance, a plot of the phase-amplitude distribution or
            None. If the distribution has not been previously calculated by
            calling this instance, this method returns None.
        """

        if self.densities is None:
            return None

        if ax is None:
            _, ax = plt.subplots()

        # by default set width of bars binsize - 2 and align on left
        width = kwargs.pop('width', self.binsize - 2)
        align = kwargs.pop('align', 'edge')
        # set the left bin edges and bar the densities
        xs = np.arange(0, 360, self.binsize)
        ax.bar(xs, self.densities, width=width, align=align, **kwargs)

        return ax

    def __call__(
        self,
        phases: npt.NDArray,
        amplitudes: npt.NDArray,
    ) -> float:
        """Returns the moduluation index for time-series' of phases
        & corresponding amplitudes.

        Args:
            phases:
                A 1-D array of phases in the interval [0, 360].
            amplitudes:
                A 1-D array of amplitudes 1 per phase in phases.

        Returns:
            The modulation index float value.
        """

        digitized = np.digitize(phases, bins=self.phase_bins)

        means = []
        for bin_idx in range(self.num_bins):
            locs = np.where(digitized == bin_idx)
            means.append(np.mean(amplitudes[locs]))

        self.densities = np.array(means) / np.sum(means)

        # compute and return the normalized KL-divergence
        log_n = np.log10(self.num_bins)
        return (log_n - numerical.shannon(self.densities)) / log_n


class VectorLength:
    """ """

    def __init__(self):
        """ """

        self.composite = None

    def plot(self, ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[None, plt.Axes]:

        """ """

        if self.composite is None:
            return None

        if ax is None:
            _, ax = plt.subplots()

        imag = np.imag(self.composite)
        real = np.real(self.composite)

        ax.plot(real, imag, marker='o', color=[.8, .8, .8], alpha=0.5)

        return ax


    def __call__(self, phases, amplitudes):
        """ """

        # would be nice to give the magnitude and angle of the mean vector
        self.composite = amplitudes * np.exp(1j * phases)
        n = len(phases)
        return np.abs(1/n * np.sum(self.composite))



if __name__ == '__main__':
    from openseize.filtering.fir import Kaiser

    from phasor.data.synthetic import PAC

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=0.8)
    # changing the duration from 3 to 2 creates openseize problem
    # FIXME !
    time, signal = pac(3, fs=500, shift=90, sigma=0.1, seed=0)

    phase_kaiser = Kaiser(
        fpass=[8, 12], fstop=[6, 14], fs=500, gpass=0.1, gstop=40
    )
    theta_signal = phase_kaiser(signal, chunksize=len(signal), axis=-1)

    amp_kaiser = Kaiser(
        fpass=[40, 60], fstop=[35, 65], fs=500, gpass=0.1, gstop=40
    )
    gamma_signal = amp_kaiser(signal, chunksize=len(signal), axis=-1)

    filtered = np.stack((theta_signal, gamma_signal))

    x = numerical.analytic(filtered, axis=-1)
    phases = numerical.phases(x[0])
    amplitudes = numerical.envelopes(x[1])

    """
    mi = ModulationIndex(num_bins=18)
    result = mi(phases, amplitudes)
    """

    fig, axarr = plt.subplots(5, 1, figsize=(6, 8))
    axarr[0].plot(time, signal, label='signal')
    axarr[1].plot(time, filtered[0], label='theta filtered')
    axarr[2].plot(time, phases, label='theta phases')
    axarr[3].plot(time, filtered[1], label='Gamma filtered')
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


    mvl = VectorLength()
    result = mvl(phases, amplitudes)
    mvl.plot()
    plt.show()
