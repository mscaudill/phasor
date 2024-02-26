"""A collection of cross-frequency coupling metrics.

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from phasor.coupling.bases import PAC
from phasor.core import numerical

# FIXME
# 3. Part of this API should consider statistical test such as trial or block
#    swapping???? 


class ModulationIndex(PAC):
    """A phase-amplitude coupling measure based on the distance between the
    observed phase-amplitude distribution and a uniform distribution.

    This is the method of Tort et. al. 2010. It constructs an observed
    phase-amplitude distribution by binning the phases and computing the average
    amplitude within each phase bin. This observed distribution is then compared
    using the Kullback-Leibler divergence to a uniform distribution. Larger
    values indicate more "suprise" when using the Uniform as a model for the
    observed distribution.

    References:
        1. Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
           phase-amplitude coupling between neuronal oscillations of different
           frequencies. J Neurophysiol. 2010 Aug; 104(2):1195-210. doi:
           10.1152/jn.00106.2010.
    """

    def __init__(self, num_bins: int):
        """Initialize this metric with a number of bins to partition the phases.

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

    def plot(self, ax=None, **kwargs):
        """Constructs a phase-amplitude distribution plot from this instance's
        denstity.

        Args:
            ax:
                A matplotlib Axes instance to plot the binned phases and mean
                amplitudes to. If None, a new figure is created.
            kwargs:
                Any valid kwarg for matplotlib's bar function.
        """

        if ax is None:
            fig, ax = plt.subplots()

        # by default set width of bars binsize - 2 and align on left 
        width = kwargs.pop('width', self.binsize-2)
        align = kwargs.pop('align', 'edge')
        # set the left bin edges and bar the densities
        xs = np.arange(0, 360, self.binsize)
        ax.bar(xs, self.densities, width=width, align=align, **kwargs)

        return ax

    def __call__(self, phases, amplitudes):
        """ """

        digitized = np.digitize(phases, bins=self.phase_bins)

        means = []
        for bin_idx in range(self.num_bins):

            locs = np.where(digitized == bin_idx)
            means.append(np.mean(amplitudes[locs]))

        self.densities = np.array(means) / np.sum(means)

        logN = np.log10(self.num_bins)
        return (logN - numerical.shannon(self.densities)) / logN



if __name__ == '__main__':

    from phasor.data.synthetic import PAC
    from phasor.core import numerical
    from phasor.coupling.__depr__.filtering import Multifilter
    from openseize.filtering.fir import Kaiser

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=.8)
    # changing the duration from 3 to 2 creates openseize problem
    # FIXME !
    time, signal = pac(3, fs=500, shift=90, sigma=0.1, seed=0)

    phase_kaiser = Kaiser(
                    fpass=[8, 12],
                    fstop=[6, 14],
                    fs=500,
                    gpass=0.1,
                    gstop=40)
    theta_signal = phase_kaiser(signal, chunksize=len(signal), axis=-1)

    amp_kaiser = Kaiser(
                    fpass=[40, 60],
                    fstop=[35, 65],
                    fs=500,
                    gpass=0.1,
                    gstop=40)
    gamma_signal = amp_kaiser(signal, chunksize=len(signal), axis=-1)

    filtered = np.stack((theta_signal, gamma_signal))

    x = numerical.analytic(filtered, axis=-1)
    phases = numerical.phases(x[0])
    amplitudes = numerical.envelopes(x[1])

    mi = ModulationIndex(num_bins=18)
    result = mi(phases, amplitudes)
    
    #mi.plot()
    #plt.show()

    fig, axarr = plt.subplots(5,1, figsize=(6,8))
    axarr[0].plot(time, signal, label='signal')
    axarr[1].plot(time, filtered[0], label='theta filtered')
    axarr[2].plot(time, phases, label='theta phases')
    axarr[3].plot(time, filtered[1], label='Gamma filtered')
    axarr[3].plot(time, amplitudes, label='Gamma Amplitude')
   
    axarr[0].sharex(axarr[1])
    axarr[1].sharex(axarr[2])
    axarr[2].sharex(axarr[3])
    #axarr[0].get_shared_x_axes().join(axarr[1], axarr[2], axarr[3])
    # show the magnitude of the Fourier spectrum
    freqs = np.fft.rfftfreq(len(signal), d=1/500)
    signal_fft = np.fft.rfft(signal)
    axarr[4].plot(freqs, np.abs(signal_fft), label='Spectrum')

    [ax.legend() for ax in axarr]
    plt.show()

    mi.plot()
    plt.show()

