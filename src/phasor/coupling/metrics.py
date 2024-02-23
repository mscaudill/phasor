"""A collection of cross-frequency coupling metrics.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt





class ModulationIndex:
    """ """

    def __init__(self, binsize):
        """ """

        self.binsize = binsize

    def _density(self, phases, amplitudes, phase_unit='deg'):
        """ """

        if phase_unit.lower() == 'deg':
            max_phase = 360
        elif phase_unit.lower() == 'rad':
            max_phase = 2 * np.pi
        else:
            raise ValueError("phase unit must be one of ['deg', 'rad']")

        bins = np.arange(self.binsize, 360, self.binsize)
        digitized = np.digitize(phases, bins=bins)

        means = []
        for phase_bin in range(len(bins)+1):

            locs = np.where(digitized == phase_bin)
            means.append(np.mean(amplitudes[locs]))

        self.bins = bins
        self.density = np.array(means) / np.sum(means)

    def plot(self):
        """ """

        width = self.bins[1]-self.bins[0] - 0.1
        xs = np.arange(0, 360, self.binsize)
        fig, ax = plt.subplots()
        ax.bar(xs, self.density, width=self.binsize-2, align='edge')
        ax.set_ylim([0, .1])


if __name__ == '__main__':

    from phasor.data.synthetic import PAC
    from phasor.core import numerical
    from phasor.coupling.__depr__.filtering import Multifilter
    from openseize.filtering.fir import Kaiser

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=.8)
    # changing the duration from 3 to 2 creates openseize problem
    # FIXME !
    time, signal = pac(3, fs=500, phi=90, sigma=0.1, seed=0)

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

    mi = ModulationIndex(binsize=20)
    mi._density(phases, amplitudes)
    
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

