"""A class for building any of Cohen's bilinear time-frequency distributions.

Bilinear TFDs represent the energy density of a signal simulataneously in time
& frequency. Unlike short-time Fourier transforms and wavelets, these joint
distributions provide high uniform resolution across both time and frequency for
non-stationary signal analysis. In particular, they are well-suited for
computing energy fractions over small time & frequency ranges, computing
freqeuncy distributions at specific time instantances, or moments of the
distribution.

$$
    P(f, t) \equiv \int\int A(\nu, \tau)K(\nu, \tau)e^{-i \nu t}e^{-i \omega
    \tau} d\nu d\tau
$$

Classes:
    Bilinear:
        A callable for constructing any of Cohen's bilinear TFDs. To construct
        a specific bilinear TFD such as Rihaczek clients may supply kernels
        using the 'add_kernel' method. Passing multiple kernels will result in
        a single kernel, the product of all supplied kernels.
"""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.linalg import toeplitz

from phasor.core.arraytools import pad_axis_to
from phasor.distributions import kernels


class Bilinear:
    """Cohen's class of Bilinear Time Frequency Distributions.

    Each of Cohen's TFDs are distinguished by a kernel choice. These kernels are
    used to reduce interference terms arising from the instantaneous
    auto-correlation (see References). The default is to use a unity kernel.
    This choice yields the famous Wigner distribution that will contain
    cross-term interference for multicomponent signals.

    Examples:
    >>> # build a signal containing noisy sine waves
    >>> from phasor.data.synthetic import MultiSine
    >>> # build two sine waves slightly shifted in time and frequency
    >>> sines = MultiSine(amps=[1, 1], freqs=[9, 11], times=[
    ... [5.5, 9.5], [6, 10]])
    >>> time, signal = sines(duration=12, fs=128, sigma=0.05, seed=None)
    >>> # build a Wigner TFD
    >>> wigner = Bilinear()
    >>> density, freqs, times = wigner(signal, fs=128)
    >>> print(density.shape) # 128 * 12 + 1
    (1537, 1537)
    >>> print(f'Time resolution = {times[1] - times[0]}')
    Time resolution = 0.0078125
    >>> print(np.ceil(max(freqs)))
    32.0
    >>> # expected frequency resolution is 128/2 / NFFT=1537
    >>> print(f'Frequency resolution = {np.round(np.abs(freqs[1] - freqs[0]), 6)}')
    Frequency resolution = 0.04164

    References:

        1. L. Cohen, "Time-frequency distributions-a review," in Proceedings of
           the IEEE, vol. 77, no. 7, pp. 941-981, July 1989, doi:
           10.1109/5.30749.
        2. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
           tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
        3. Cohen, L. (1995) Time-Frequency Analysis. Prentice-Hall Signal Processing
        4. H.I. Choi and W. J. Williams, "Improved time-frequency
           representation of multicomponent signals using exponential kernels,"
           in IEEE Transactions on Acoustics, Speech, and Signal Processing,
           vol. 37, no. 6, pp. 862-871, June 1989, doi: 10.1109/ASSP.1989.28057.
        5. R. M. Fano, “Short-timeautocorrelation functions and power
           spectra,” J. Acoust. Soc. Am., vol. 22, pp. 546-550, 1950.
    """

    def __init__(self, detrend=True, analytic=True) -> None:
        """Initialize this TFD.

        Args:
            detrend:
                A boolean indicating if the signal should be detrended. If
                signal has a non-zero mean it should be detrended.
            analytic:
                A boolean indicating if the negative frequency components should
                be removed. This prevents negative frequency aliasing in the
                time-frequency distributions.
        """

        self.detrend = detrend
        self.analytic = analytic
        self.kernels = []

    def domain(
            self,
            signal: npt.NDArray,
            fs: float,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns the doppler frequencies & lag coordinates of the ambiguity
        domain.

        The ambiguity function transforms the instantaneous autocorrelation from
        a function of time & delay to a function of doppler frequency shifts
        & doppler lags. This function returns these ambiguity domain vectors.

        Args:
            signal:
                A 1-D numpy array of signal values in time.
            fs:
                The sampling rate of the signal in Hz.

        Returns:
            A 2-tuple of 1-D arrays of doppler frequencies & lags. The doppler
            frequencies have shape length(signal) + 1 & are returned in the
            order specified by numpy's fftfreq function. The doppler delays
            range from -2l/fs to 2l/fs where l is an integer shift of the
            signal. For odd lengthed signals the abs(l) < (len(signal) // 2 - 1)
            & for even-lengthed signals abs(l) < len(signal) // 2.
        """

        etas = np.fft.fftfreq(len(signal), d=1/fs)
        # max_shift depends on even/odd signal length
        max_shift = int(np.ceil(len(signal) / 2) - 1)
        taus = 2 * np.arange(-max_shift, max_shift+1) / fs

        return etas, taus

    def add_kernel(self, kernel, **kwargs):
        """Add a kernel to this time-frequency distribution.

        Kernel functions are key to building useful time-frequency distributions
        as they mask the doppler-lag domain of the ambiguity function. They are
        useful in attenuating cross-term interference of multicomponent signals.

        Args:
            kernel:
                A callable that must accept the doppler frequency and lag
                variable. Please see phasor.distributions.kernels
            **kwargs:
                Any keyword arguments needed to specify a kernel excluding the
                doppler frequencies (etas) and lags (taus) positional arguments.

        Returns:
            None but stores a partial function of the kernel callable in which
            all parameters except the doppler-lag variables have been frozen.
        """

        # pop etas and taus if client passed them
        [kwargs.pop(x, None) for x in {'etas', 'taus'}]
        func = partial(kernel, **kwargs)
        self.kernels.append(func)

    def _autocorrelation(self, signal: npt.NDArray) -> npt.NDArray:
        """Returns the instantaneous autocorrelation (AC) for a 1-D signal.

        Following reference 2, the time-dependent AC is defined as:

            AC = x^(t - l) * x(t + l)

        where x is the signal, ^ denotes complex conjugation & l is an integer
        shift of the signal. The AC is a function of both time & shift.  Note
        the shift is 1/2 the delay in in the AC definintion (see Refs 2 & 3).

        Args:
            signal:
                A 1-D numpy array whose AC is to be measured.

        Returns:
            A 2-D array of autocorrelations with delays along axis 0
            & time along axis 1.

        Notes:
            We compute the autocorrelation using Toeplitz matrices that hold both
            the forward and reverse signal copies. Consider this signal [7,4,6].
            Its lag can range from -3 to +3. Here are the reversed & forward
            delayed signals.

             delay  s^[n-l]   s[n+l]   Autocorr.
            -3      [0,0,0]   [0,0,0]  [0,0,0]
            -2      [6,0,0]   [0,0,7]  [0,0,0]
            -1      [4,6,0]   [0,7,4]  [0,42,0]
             0      [7,4,6]   [7,4,6]  [49,16,36]
             1      [0,7,4]   [4,6,0]  [0,42,0]
             2      [0,0,7]   [6,0,0]  [0,0,0]
             3      [0,0,0]   [0,0,0]  [0,0,0]

            Non-zero products occur at delays = {-1,0,1} & the signals are
            Toeplitz and flipped Toeplitz matrices respectively.

        References:
            1. R. M. Fano, “Short-timeautocorrelation functions and power
               spectra,” J. Acoust. Soc. Am., vol. 22, pp. 546-550, 1950.
            2. L. Cohen, "Time-frequency distributions-a review," in Proceedings
               of the IEEE, vol. 77, no. 7, pp. 941-981, July 1989, doi:
               10.1109/5.30749.
            3. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
               tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
        """

        x = np.array(signal)
        if self.detrend:
            x -= np.mean(x)

        if self.analytic:
            x = sps.hilbert(x)

        # make Toeplitzes with max val of l depending on even/odd signal length
        max_l = int(np.ceil(len(signal) / 2) - 1)
        first_col = pad_axis_to(x[max_l::-1], 2 * max_l + 1, side='right')
        first_row = pad_axis_to(x[max_l:], len(x), side='right')
        reverse = toeplitz(first_col, first_row)
        forward = np.flip(reverse, axis=0)

        return np.conjugate(reverse) * forward

    def ambiguity(self, signal: npt.NDArray) -> npt.NDArray:
        """Returns the 2-D ambiguity matrix of a 1-D data array.

        The integrand of Cohen's classes of TFDs can be written in terms of an
        ambiguity function; a transform of the time-dependent autocorrelation
        to the ambiguity plane. The utility of expressing a TFD in terms of the
        ambiguity function is that it allows for careful shapping of kernel
        functions for removal of cross terms inherent in all bilinear TFDs.

        Args:
            signal:
                A 1-D signal that may be real or complex.

        Returns:
            An ambiguity  matrix with doppler delays (taus) along axis=0 &
            frequency shifts (etas) along axis=1. The shape will match the
            lengths of taus and etas described in the domain method.

        References:
            1. L. Cohen, "Time-frequency distributions-a review," in Proceedings of
               the IEEE, vol. 77, no. 7, pp. 941-981, July 1989,
               doi: 10.1109/5.30749.
        """

        # compute autocorrelation and ambiguity
        autocorr = self._autocorrelation(signal)
        return  np.fft.ifft(autocorr, axis=1)

    def __call__(
            self,
            signal: npt.NDArray,
            fs: float,
            **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns the energy density distribution & time-frequency coordinates.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.
            kwargs:
               Keyword arguments are passed to kernel method.

        Returns:
            A 3-tuple of ndarrays; a 2-D array of TFD energy densities, a 1-D
            array of frequencies from -fs/4 to fs/4 and a 1-D array of times
            from 0 to len(signal).

        References:
            1. L. Cohen, "Time-frequency distributions-a review," in Proceedings
               of the IEEE, vol. 77, no. 7, pp. 941-981, July 1989, doi:
               10.1109/5.30749.
            2. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
               tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
            3. Cohen, L. (1995) Time-Frequency Analysis. Prentice-Hall Signal
               Processing

        Notes:
            If this distribution was not assigned a kernel or a unity kernel was
            the only kernel given, this bilinear instance becomes a Wigner TDF
            that can be quickly computed without the ambiguity (see reference 2).
        """

        if not self.kernels:
            self.add_kernel(partial(kernels.unitary))

        # build a kernel matrix over the ambiguity domain
        etas, taus = self.domain(signal, fs)
        kernel = np.prod(np.stack([k(etas, taus) for k in self.kernels]), axis=0)

        # if unity kernel (i.e. Wigner TFD) use faster single fft
        if np.any(kernel - 1):
            amb = self.ambiguity(signal)
            result = 2 * np.fft.fft2(kernel * amb)
        else:
            autocorr = self._autocorrelation(signal)
            result = np.fft.fft(autocorr, axis=0)

        times = np.arange(0, len(signal)/fs, 1/fs)
        freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

        # place 0 frequency at center
        result = np.fft.fftshift(result, axes=0)
        freqs = np.fft.fftshift(freqs)

        return result, freqs, times

    def plot(
            self,
            tfd: npt.NDArray,
            freqs: npt.NDArray,
            time: npt.NDArray,
            ax: Optional[plt.Axes] = None,
    ) -> None:
        """Plots the magnitude of a time-frequency distribution.

        tfd:
            A 2-D ndarray of signal energy densities with frequencies along axis
            0 and time along axis 1.
        freqs:
            A 1-D array of frequencies in Hz.
        time:
            A 1-D array of times in secs.
        ax:
            a matplotlib axis where this plot will be shown. If None a new axis
            will be created.
        """

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(time, freqs, np.abs(tfd), shading='nearest')
        fig.colorbar(mesh, ax=ax)
        plt.show()


if __name__ == '__main__':


    from phasor.data.synthetic import MultiSine
    from phasor.distributions import kernels

    from functools import partial

    msine = MultiSine(
                amps=[1,1],
                freqs=[9, 11],
                times=[[5.5, 9.5], [6, 10]])
    time, signal = msine(duration=12, fs=128, sigma=0.01, seed=None)

    
    wigner = Bilinear()
    #wigner.add_kernel(kernels.choi_williams, sigma=0.1)
    tfd, freqs, times = wigner(signal, fs=128)
    wigner.plot(tfd, freqs, times)
    

    """
    bilinear = Bilinear()
    bilinear.add_kernel(kernels.unitary)
    bilinear.add_kernel(kernels.choi_williams, width=0.1)
    cw = partial(kernels.choi_williams, width=0.1)
    tfd, freqs, times = bilinear(signal, fs=128)
    bilinear.plot(tfd, freqs, times)
    """

    """
    rihaczek = Bilinear()
    rihaczek.add_kernel(kernels.rihaczek)
    rihaczek.add_kernel(kernels.choi_williams, width=0.1)
    tfd, freqs, times = rihaczek(signal, fs=128, width=0.1)
    rihaczek.plot(tfd, freqs, times)
    """
