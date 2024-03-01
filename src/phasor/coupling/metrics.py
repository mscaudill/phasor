"""A collection of cross-frequency coupling metrics including:

    - ModulationIndex: A phase-amplitude coupling metric that forms an observed
      phase-amplitude distribution and measures its distance from a uniform
      distribution (Torte et al. 2010).
    - MeanVectorLength: A phase-amplitude coupling metric that measures the mean
      vector length of a complex composite signal formed from the amplitudes and
      phases (Canolty et al. 2006).
    - PhaseLockingIndex: A phase-phase coupling metric that measures the
      conditional probability of one phase given the other (Tass et al. 1998).

References:
    1. Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
       phase-amplitude coupling between neuronal oscillations of different
       frequencies. J Neurophysiol. 2010 Aug; 104(2):1195-210. doi:
       10.1152/jn.00106.2010.
    2. Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
       Berger MS, Barbaro NM, Knight RT. High gamma power is phase-locked to
       theta oscillations in human neocortex. Science. 2006 Sep
       15;313(5793):1626-8. doi: 10.1126/science.1128115
    3. Tass P., Rosenblum M., Weule J., Kurths J., Pikovsky A., Volkmann J.
       Detection of n:m phase locking from noisy data: application to
       magnetoencephalography. Phys Rev Lett. 1998;81(15):3291–3294.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from phasor.core import numerical


class ModulationIndex:
    """A phase-amplitude coupling metric based on the distance between the
    observed phase-amplitude distribution and a uniform distribution.

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

        ax.set_xlabel('Phase (deg)')
        ax.set_ylabel('Amplitude')

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


class MeanVectorLength:
    """A phase-amplitude coupling metric callable that measures the mean vector
    length of a sequence of complex vectors encoding amplitude and phase.

    This is the method of Canolty et. al. 2006. It constructs a time series of
    complex vectors A(t) * exp(i * phi(t)) from the amplitude time series A(t)
    and the phase time series phi(t). The mean length of these complex vectors
    is taken as a measure of the coupling between the phases and amplitudes.
    Intuitively, if there is a relationship between phases & amplitudes, this
    will create an asymmetry in the distribution of these vectors in the complex
    plane. This may be visualized with this intance's plot method.

    Attributes:
        composite:
            A 1-D array of complex vector values that are added to this instance
            following a call invocation.

    References:
        1. Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE,
           Berger MS, Barbaro NM, Knight RT. High gamma power is phase-locked to
           theta oscillations in human neocortex. Science. 2006 Sep
           15;313(5793):1626-8. doi: 10.1126/science.1128115
    """

    def __init__(self):
        """Initialize this instance with None for the composite signal."""

        self.composite = None

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[None, plt.Axes]:
        """Plots the time series of complex vectors of the composite signal to
        the imaginary plane.

        Args:
            ax:
                A matplotlib Axes for plotting the complex vectors to. If None,
                a new figure & axis is created.
            kwargs:
                Any valid kwarg for matplotlib's plot function.

        Returns:
            An Axes instance, a plot of the complex vectors or
            None. If the vectors have not been previously calculated by
            calling this instance, this method returns None.
        """

        if self.composite is None:
            return None

        if ax is None:
            _, ax = plt.subplots()

        # get the imaginary, real and mean vectors of the composite signal
        imag = np.imag(self.composite)
        real = np.real(self.composite)
        mean_vector = np.mean(self.composite)

        # set reasonable defaults for lines
        color = kwargs.pop('color', [0.9, 0.9, 0.9])
        linewidth = kwargs.pop('linewidth', 1)
        # set reasonable defaults for markers
        marker = kwargs.pop('marker', 'o')
        msize = kwargs.pop('markersize', 4)
        markerfacecolor = kwargs.pop('markerfacecolor', 'k')

        # construct plot and add arrow for mean vector
        ax.plot(
            real,
            imag,
            marker=marker,
            markerfacecolor=markerfacecolor,
            markersize=msize,
            color=color,
            linewidth=linewidth,
            **kwargs,
        )
        ax.quiver(0, 0, np.real(mean_vector), np.imag(mean_vector), angles='xy')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')

        return ax

    def __call__(self, phases, amplitudes):
        """Returns the length and angle of the mean composite vector.

        Args:
            phases:
                A 1-D array of phases in the interval [0, 360].
            amplitudes:
                A 1-D array of amplitudes 1 per phase in phases.

        Returns:
            A 2-tuple containing the mean vector length and its angle in
            degrees.
        """

        # construct the composite and compute its mean
        self.composite = amplitudes * np.exp(1j * phases * np.pi / 180)
        mean_vector = np.mean(self.composite)

        magnitude = np.abs(mean_vector)
        angle = np.angle(mean_vector, deg=True)
        return magnitude, angle


class PhaseLockIndex:
    """A phase-phase coupling metric that assess the conditional probability of
    observing a phase given that the other phase lies within a binned range.

    This is the method of Tass et. al. 1998 (see reference 1). This index is
    constructed following the following steps:
        1. given two phase series phi_0 and phi_1, digitize phi_0 into bins
        2. For a phi_0 bin measure v = <exp(i * phi_1(j))> where j are samples
           where phi_0 lies in this bin and < > is the expectation operator.
        3. Compute the magnitude of v for each bin. This |v| is in [0, 1]. If
           there is a perfect phase-phase relationship |v| = 1 and if there is
           no relationship |v| ~ 0 within this bin.
        4. Finally, average the magnitudes across bins. This represents the
           conditional probability of observing phi_1 given phi_0.

    Attributes:
        num_bins:
            The number of bins used to partition the phases which are assumed to
            be in degree units [0, 360].
        binsize:
            The size of each bin in degrees.
        phase_bins:
            A 1-D array of right-side bin edges in [binsize, 360].
        vectors:
            A 1-D array of mean complex exponential vectors one per bin in
            num_bins. This attribute is initialized None & set to a 1-D array
            after this instance is called on a set of phases.

    References:
        1. Tass P., Rosenblum M., Weule J., Kurths J., Pikovsky A., Volkmann J.
           Detection of n:m phase locking from noisy data: application to
           magnetoencephalography. Phys Rev Lett. 1998;81(15):3291–3294.
    """

    def __init__(self, num_bins: int):
        """Initialize this PhaseLockIndex metric.

        Args:
            num_bins:
                An integer number of bins for partitioning phases.

        Returns:
            None
        """

        self.num_bins = num_bins
        self.binsize = 360 / self.num_bins
        # bins are defined by their largest (i.e. rightmost) values
        self.phase_bins = np.linspace(self.binsize, 360, self.num_bins)
        self.vectors = None

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[None, plt.Axes]:
        """Constructs a polar bar plot of the magnitude of the complex
        exponential vectors one per bin in this instances phase bins.

        Args:
            ax:
                A matplotlib Axes using a polar projection. If None, a
                figure & axis are created.
            kwargs:
                Any valid kwarg for matplotlib's bar function.

        Returns:
            An Axes instance, a bar plot of the mean complex vector magnitudes
            in each phase bin. If the vectors have not been previously
            calculated by calling this instance, this method returns None.
        """

        if self.vectors is None:
            return None

        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        phases = self.phase_bins * np.pi / 180
        magnitudes = np.abs(self.vectors)
        # configure plot with reasonable defaults
        widths = kwargs.pop('widths', 2 * np.pi / self.num_bins)
        align = kwargs.pop('align', 'edge')
        bottom = kwargs.pop('bottom', 0)
        edgecolor = kwargs.pop('edgecolor', 'white')
        title = 'Phase-Phase Coupling Per Bin'

        ax.bar(
            phases,
            magnitudes - bottom,
            width=widths,
            align=align,
            bottom=bottom,
            edgecolor=edgecolor,
            **kwargs,
        )
        ax.set_xlabel('Phase (deg)')
        ax.set_title(title)

        return ax

    def __call__(
        self,
        phases: npt.NDArray,
        others: npt.NDArray,
        n: int = 1,
        m: int = 1,
    ) -> float:
        """Returns the phase-locking index, the conditional probability of
        observing a specific phase given that the other phase lies within
        a binned range.

        Args:
            phases:
                A 1-D array of phases in the interval [0, 360].
            others:
                A 1-D array of phases in the interval [0, 360].
            n, m:
                The locking condition integers defined by the locking condition:
                n * phase - m * other < |epsi| where epsi is usually on the
                order of the noise magnitude. E.g if n=1 & m=2 then perfect
                locking means phases are 2x the phase values of others. The
                default case 1:1, measures the 0 phase difference
                synchronization.

        Returns:
            A float phase-locking conditional probability in [0, 1] where
            0 indicates no phase-other relationship and 1 indicates perfect
            n:m synchronization.
        """

        x = np.mod(n * phases, 360)
        y = np.mod(m * others, 360)
        digitized = np.digitize(x, bins=self.phase_bins)

        self.vectors = []
        # compute mean exponential vectors of others for each phase bin
        for bin_idx in range(self.num_bins):
            locs = np.where(digitized == bin_idx)
            vectors = np.exp(1j * y[locs] * np.pi / 180)
            self.vectors.append(np.mean(vectors))

        return np.mean(np.abs(self.vectors))


if __name__ == '__main__':
    from openseize.filtering.fir import Kaiser

    from phasor.data.synthetic import PAC

    pac = PAC(fp=10, fa=50, amp_p=1.8, amp_a=1, strength=0.8)
    # changing the duration from 3 to 2 creates openseize problem
    # FIXME !
    time, signal = pac(3, fs=500, shift=0, sigma=0.1, seed=0)

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
    """

    # plot the modulation index
    """
    fig2, ax = plt.subplots()
    mi = ModulationIndex(num_bins=18)
    mi_result = mi(phases, amplitudes)
    mi.plot(ax)
    """

    """
    mvl = VectorLength()
    mvl_result = mvl(phases, amplitudes)
    mvl.plot()
    plt.ion()
    plt.show()
    """

    # get the phases of the gamma envelope
    others = numerical.phases(numerical.analytic(gamma_signal))
    # others = np.random.randint(0, 360, len(signal))

    pli = PhaseLockIndex(num_bins=18)
    pli_result = pli(phases, others, n=1, m=1)
    pli.plot()
    plt.show()
