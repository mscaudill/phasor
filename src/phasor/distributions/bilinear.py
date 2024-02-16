"""A collection of classes for building Cohen's bilinear time frequency
distributions.

Bilinear TFDs represent the energy density of a signal simulataneously in time
and frequency. Unlike short-time Fourier transforms and Wavelets, these joint
distributions provide high uniform resolution across both time and frequency for
non-stationary signal analysis. In particular, they are well-suited for
computing energy fractions over small time/frequency ranges, computing freqeuncy
distributions at specific time instantances, or moments of the distribution.

Classes:
    Bilinear:
        A callable for constructing any of Cohen's bilinear TFDs. This callable
        defaults to the Choi-Williams kernel for reducing interference terms.
        Other TFDs in this module (Wigner, Rihaczek, etc) inherit Bilinear and
        override the kernel to implement their specific TFDs.
    Wigner:
        A callable for constructing the Wigner TFD. It has a kernel of 1 and
        overrides Bilinear's call method to since this TFD does not require
        the ambiguity function. A consequence of a unitary kernel is that
        cross-term interference will be present for multicomponent signals.
    Rihaczek:
        A callable for constructing a complex Rihaczek TFD. It uses a kernel of
        1 and therefore has cross-term interference for multicomponents signals.
    RID_Rihaczek:
        A callable for constructing a reduced interference complex Rihaczek TFD.
        To reduce cross-term contributions it uses a Choi-Williams kernel. The
        advantage of a complex TFD is the extraction of phase information within
        specified frequency ranges.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.linalg import toeplitz

from phasor.core.arraytools import pad_axis_to


class Bilinear:
    """Cohen's class of Bilinear Time Frequency Distributions.

    Each of Cohen's TFDs are distinguished by a kernel choice. These kernels are
    used to reduce interference terms arising from the instantaneous
    auto-correlation (see References). The default kernel is the Choi-Williams
    kernel. Specific TFDS may be constructed by inheriting this class and
    overridding the kernel choice (see Wigner, Rihaczek etc.).

    Examples:
    >>> 

    References:

        1. L. Cohen, "Time-frequency distributions-a review," in Proceedings of
           the IEEE, vol. 77, no. 7, pp. 941-981, July 1989, doi:
           10.1109/5.30749.
        2. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
           tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
        3. Cohen, L. (1995) Time-Frequency Analysis. Prentice-Hall Signal Processing
        4. H. . -I. Choi and W. J. Williams, "Improved time-frequency
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

    def ambiguity(self, signal: npt.NDArray, fs: float) -> npt.NDArray:
        """Returns the 2-D ambiguity matrix of a 1-D data array.

        The integrand of Cohen's classes of TFDs can be written in terms of an
        ambiguity function; a transform of the time-dependent autocorrelation
        to the ambiguity plane. The utility of expressing a TFD in terms of the
        ambiguity function is that it allows for careful shapping of kernel
        functions for removal of cross terms inherent in all bilinear TFDs.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.

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

    def kernel(
            self,
            signal: npt.NDArray,
            fs: float,
            width: float,
    ) -> npt.NDArray:
        """The Choi-Williams kernel for cross-term reduction.

        Bilinear TFDs will have cross-terms for multicomponent signals. In the
        ambiguity plane these cross-terms tend away from the origin while the
        auto-terms map near the origin. Thus kernels like this one low-pass
        filter around the origin in the ambiguity plane to reduce the
        cross-term contiribution.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.
            width:
                The dispersion of the kernel about (eta=0, tau=0).

        Returns:
            A 2-D matrix of ones of shape lags x doppler frequencies in the
            ambiguity domain (see domain).
        """

        etas, taus = self.domain(signal, fs)
        cols, rows = np.meshgrid(etas, taus)
        return np.exp(-(cols**2 * rows**2) / width)

    def __call__(
            self,
            signal: npt.NDArray,
            fs: float,
            **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns the energy density distribution and corresponding
        time-frequency coordinates.

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
        """

        amb = self.ambiguity(signal, fs)
        k = self.kernel(signal, fs, **kwargs)
        result = 2 * np.fft.fft2(k * amb)

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


class Wigner(Bilinear):
    """The Wigner bilinear time-frequency distribution.

    The Wigner TFD is equivalent to a Cohen's TFD with a kernel of 1. It thus
    retains cross-term interference for multicomponent signals.
    """

    def kernel(signal, fs):
        """The Wigner kernel is a matrix of ones across the ambiguity
        coordinates eta and tau.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.

        Returns:
            A 2-D matrix of ones of shape lags x doppler frequencies in the
            ambiguity domain (see domain).
        """

        etas, taus = self.domain(signal, fs)
        cols, rows = np.meshgrid(etas, taus)
        return np.ones((rows, cols))

    def __call__(self, signal, fs):
        """Returns the energy density distribution and corresponding
        time-frequency coordinates for the Wigner TFD.

        With a kernel choice of 1 the Wigner can be calculated with a single
        Fourier transform of the lag variable tau (see Reference 1), so we
        override Bilinear's call method.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.

        Returns:
            A 3-tuple of ndarrays; a 2-D array of TFD energy densities, a 1-D
            array of frequencies from -fs/4 to fs/4 and a 1-D array of times
            from 0 to len(signal).
        """


        autocorr = self._autocorrelation(signal)
        result = np.fft.fft(autocorr, axis=0)

        times = np.arange(0, len(signal)/fs, 1/fs)
        freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

        # place 0 frequency at center
        result = np.fft.fftshift(result, axes=0)
        freqs = np.fft.fftshift(freqs)

        return result, freqs, times


class Rihaczek(Bilinear):
    """The Rihaczek complex bilinear TFD.

    To obtain phase information in a TFD, the TFD must be complex. Using the
    Rihaczek kernel k(eta, tau) = i * (eta * tau) / 2, Cohen's bilinear TFD
    becomes complex. As with all bilinear transforms it suffers from cross-term
    interference when the signal is multicomponent.
    """

    def kernel(signal, fs):
        """The smoothing kernel is a matrix of ones across the ambiguity
        coordinates eta and tau.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.

        Returns:
            A 2-D matrix of ones of shape lags x doppler frequencies in the
            ambiguity domain (see domain).
        """

        etas, taus = self.domain(signal, fs)
        cols, rows = np.meshgrid(etas, taus)
        return np.ones((rows, cols))

    def __call__(
            self,
            signal: npt.NDArray,
            fs: float,
            **kwargs
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns the complex Rihaczek energy density distribution
        & corresponding time-frequency coordinates.

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
        """

        # compute the Rihaczek kernel
        etas, taus = self.domain(signal, fs)
        cols, rows = np.meshgrid(etas, taus)
        rihaczek_kernel = np.exp(1j * (cols * rows) / 2)

        # compute the bilinear TFD from the kernel products and amb
        amb = self.ambiguity(signal, fs)
        k = self.kernel(signal, fs, **kwargs)
        result = 2 * np.fft.fft2(k * rihaczek_kernel * amb)

        times = np.arange(0, len(signal)/fs, 1/fs)
        freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

        # place 0 frequency at center
        result = np.fft.fftshift(result, axes=0)
        freqs = np.fft.fftshift(freqs)

        return result, freqs, times


class RID_Rihaczek(Rihaczek):
    """ """

    def kernel(
            self,
            signal: npt.NDArray,
            fs: float,
            width: float,
    ) -> npt.NDArray:
        """The Choi-Williams kernel for cross-term reduction.

        Bilinear TFDs will have cross-terms for multicomponent signals. In the
        ambiguity plane these cross-terms tend away from the origin while the
        auto-terms map near the origin. Thus kernels like this one low-pass
        filter around the origin in the ambiguity plane to reduce the
        cross-term contiribution.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.
            width:
                The dispersion of the kernel about (eta=0, tau=0).

        Returns:
            A 2-D matrix of ones of shape lags x doppler frequencies in the
            ambiguity domain (see domain).
        """

        return super(Rihaczek, self).kernel(signal, fs, width)

if __name__ == '__main__':


    from phasor.data.synthetic import MultiSine

    msine = MultiSine(
                amps=[1,1,1],
                freqs=[11, 9, 18],
                times=[[6, 10], [6, 10], [2,5]])
    time, signal = msine(duration=12, fs=128, sigma=0.01, seed=None)

    """
    wigner = Wigner()
    tfd, freqs, times = wigner(signal, fs=128)

    wigner.plot(tfd, freqs, times)
    """

    """
    bilinear = Bilinear()
    tfd, freqs, times = bilinear(signal, fs=128, width=0.1)
    bilinear.plot(tfd, freqs, times)
    """

    rihaczek = RID_Rihaczek()
    tfd, freqs, times = rihaczek(signal, fs=128, width=0.1)
    rihaczek.plot(tfd, freqs, times)

