"""

"""

from collections import abc
from abc import abstractmethod
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.linalg import hankel, toeplitz

from phasor.core.arraytools import pad_axis_to


class QTFD(abc.Callable):
    """ """

    def __init__(self, detrend=True, analytic=True) -> None:
        """Initialize this Cohen's Quadratic Time Frequency distribution.

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

    def _autocorrelation(
            self,
            signal: npt.NDArray,
    )-> npt.NDArray:
        """Returns the time-dependent autocorrelation (ACF) for a 1-D signal.

        Following reference 2, the time-dependent ACF is defined as:

            ACF = x^(t - l) * x(t + l)

        where x is the signal, ^ denotes complex conjugation & l is an integer
        lag (i.e. the shift amounti). It is a function of both time and lag.
        Note the lag is 1/2 the delay in in the ACF definintion (see Reference
        1, 3).

        Args:
            signal:
                A 1-D numpy array whose ACF is to be measured.

        Returns:
            A 2-D array of autocorrelation values with signal delay along
            the 0th axis and time along the 1st axis.

        Notes:
            We compute the autocovariance using Toeplitz matrices that hold both
            the forward and reverse signal copies. Consider this signal [7,4,6].
            Its lag can range from -3 to +3. Here are the reversed & forward
            delayed signals.

             delay  Reversed  Forward  Autocorr.
            -3      [0,0,0]   [0,0,0]  [0,0,0]
            -2      [6,0,0]   [0,0,7]  [0,0,0]
            -1      [4,6,0]   [0,7,4]  [0,42,0]
             0      [7,4,6]   [7,4,6]  [49,16,36]
             1      [0,7,4]   [4,6,0]  [0,42,0]
             2      [0,0,7]   [6,0,0]  [0,0,0]
             3      [0,0,0]   [0,0,0]  [0,0,0]

            Notice non-zero products occur at delays = {-1,0,1} and the Reversed and
            Forward signals are a Toeplitz and a flipped Toeplitz matrix
            respectively.

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
        if x.ndim != 1:
            msg = 'data must be exactly 1-D.'
            raise ValueError(msg)

        if self.detrend:
            x -= np.mean(x)

        if self.analytic:
            x = sps.hilbert(x)

        # Construct Toeplitz
        # max_shift depends on even/odd signal length
        max_shift = len(x) // 2 if len(x) % 2 else len(x) // 2 - 1
        first_col = pad_axis_to(x[max_shift::-1], 2 * max_shift + 1, side='right')
        first_row = pad_axis_to(x[max_shift:], len(x), side='right')
        reverse = toeplitz(first_col, first_row)
        forward = np.flip(reverse, axis=0)

        return np.conjugate(reverse) * forward

    def ambiguity(self, signal: npt.NDArray, fs: float) -> npt.NDArray:
        """Returns the 2-D ambiguity matrix of a 1-D data array.

        The integrand of Cohen's classes of QTFD's can be written in terms of an
        ambiguity function; a transform of the time-dependent autocorrelation
        to the ambiguity plane. The utility of expressing a QTFD in terms of the
        ambiguity function is that it allows for careful shapping of kernel
        functions for removal of cross terms inherent in all QTFDs.

        Args:
            signal:
                A 1-D signal that may be real or complex.
            fs:
                The sampling rate of signal array.

        Returns:
            A 3-tuple consisting of:
            1. An ambiguity  matrix with doppler delays (taus) along axis=0 and
               frequency shifts (etas) along axis=1.
            2. A 1-D array of doppler delays in [-2l/fs,...2l/fs] where l is an
               integer shift of the signal. The maximum l is the largest l for
               which l < len(signal) // 2 - 1 for odd-lengthed singals and
               l < len(signal) // 2.
            3. A 1-D array of len(signal) + 1 doppler frequencies. The order of these
               frequencies is given by numpy.fft.fftfreq. Numpy's fft shift can be used
               to center these frequencies and the ambiguity matrix about the zero
               frequency for plotting.

        References:
            1. L. Cohen, "Time-frequency distributions-a review," in Proceedings of
               the IEEE, vol. 77, no. 7, pp. 941-981, July 1989,
               doi: 10.1109/5.30749.
        """

        # compute autocorrelation and ambiguity
        autocorr = self._autocorrelation(signal)
        amb = np.fft.ifft(autocorr, axis=1)

        # compute doppler shifts and lags (eta and tau respectively)
        etas = np.fft.fftfreq(len(signal), d=1/fs)
        # taus are 2X the integer shifts in ambiguity defn.
        max_shift = (amb.shape[0] - 1) // 2
        taus = 2 * np.arange(-max_shift, max_shift+1) / fs
        return amb, taus, etas

    @abstractmethod
    def __call__(self, signal, fs, kernel, *args, **kwargs):
        """Returns a tuple containing a specific time-frequency distribution,
        time vector and frequency vector."""

    def plot(self, tfd, freqs, time):
        """ """

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(time, freqs, np.abs(tfd), shading='nearest')
        fig.colorbar(mesh, ax=ax)
        plt.show()


class Wigner(QTFD):
    """ """

    def __call__(self, signal, fs, kernel=None, *args, **kwargs):
        """ """

        if np.any(kernel):
            amb, taus, etas = self.ambiguity(signal, fs=fs)
            #smoother = kernel(etas, taus)
            result = 2 * np.fft.fft2(kernel * amb)
        else:
            # bypass ambiguity when no smoothing kernel
            autocorr = self._autocorrelation(signal)
            result = np.fft.fft(autocorr, axis=0)

        times = np.arange(0, len(signal)/fs, 1/fs)
        freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

        # place 0 frequency at center
        result = np.fft.fftshift(result, axes=0)
        freqs = np.fft.fftshift(freqs)

        return result, freqs, times


if __name__ == '__main__':


    from phasor.data.synthetic import MultiSine

    msine = MultiSine(
                amps=[1,1,1],
                freqs=[4.5, 9, 12],
                times=[[6, 10], [6, 10], [2,5]])
    time, signal = msine(duration=12, fs=128, sigma=0.01, seed=None)

    def choi_williams(etas, taus, sigma):
        """ """

        cols, rows = np.meshgrid(etas, taus)
        return np.exp(-(cols**2 * rows**2) / sigma)

    # FIXME
    # the design has problems, we have to call ambiguity 2x here just to get the
    # ambiguity plane variables eta and tau for constructing our CW kernel!
    # the kernel passed to call was changed from a callable to a matrix!
    wigner = Wigner()
    amb, taus, etas = wigner.ambiguity(signal, fs=128)
    kernel = choi_williams(etas, taus, sigma=0.1)
    tfd, freqs, times = wigner(signal, fs=128, kernel=kernel)

    wigner.plot(tfd, freqs, times)


