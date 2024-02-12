"""

"""

from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import scipy.signal as sps
from scipy.linalg import hankel, toeplitz

from phasor.core.arraytools import pad_axis_to


def autocovariance(
        signal: npt.NDArray,
        detrend: bool = True,
        analytic: bool = True,
)-> npt.NDArray:
    """Returns the non-stationary autocovariance for a 1-D signal.

    The autocovariance for a non-stationary signal is:

        ACF = x^(t - l) * x(t + l)

    where x is the signal, ^ denotes complex conjugation & l is an integer lag
    (i.e. the shift amounti). It is a function of both time and lag. Note the
    lag is 1/2 the delay in in the ACF definintion (see Reference 1).

    Args:
        signal:
            A 1-D numpy array whose autocorrelation is returned.
        detrend:
            A boolean indicating if the signal should be detrended. If signal
            has a non-zero mean it should be detrended.
        analytic:
            A boolean indicating if the negative frequency components should be
            removed. This prevents negative frequency aliasing in the
            time-frequency distributions.

    Returns:
        A 2-D array of autocovariance values with signal delay along the 0th
        axis and time along the 1st axis.

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
        1. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
           tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
        2. https://en.wikipedia.org/wiki/Autocorrelation
    """

    x = np.array(signal)
    if x.ndim != 1:
        msg = 'data must be exactly 1-D.'
        raise ValueError(msg)

    if detrend:
        x -= np.mean(x)

    if analytic:
        x = sps.hilbert(x)

    # Construct Toeplitz
    # max_shift depends on even/odd signal length
    max_shift = len(x) // 2 if len(x) % 2 else len(x) // 2 - 1
    first_col = pad_axis_to(x[max_shift::-1], 2 * max_shift + 1, side='right')
    first_row = pad_axis_to(x[max_shift:], len(x), side='right')
    reverse = toeplitz(first_col, first_row)
    forward = np.flip(reverse, axis=0)

    return np.conjugate(reverse) * forward


def ambiguity(signal: npt.NDArray, fs: float, **kwargs) -> npt.NDArray:
    """Returns the 2-D ambiguity matrix of a 1-D data array.

    The ambiguity function is the short-time Fourier transform of a signal with
    a delayed copy of itself. It provides the distortion of a signal in terms of
    this delay and a doppler frequency shift. Time frequency distributions are
    built by mapping the signal to the ambiguity (delay, frequency shift) plane
    where the response is filtered before Fourier transforming back to time and
    frequency space.

    Args:
        signal:
            A 1-D signal that may be real or complex.
        fs:
            The sampling rate of signal array.
        kwargs:
            Keyword args are passed to autocorrelation.

    Returns:
        A 3-tuple consisting of:
        (1) An ambiguity  matrix with doppler delays (taus) along axis=0 and
        frequency shifts (etas) along axis=1.
        (2) A 1-D array of doppler delays. For even-lengthend signals these
        delays will be in [-2l/fs,...2l/fs] where l < len(signal) // 2 - 1. For odd
        lengthend signals these delays will be in [-2l/fs,...2l/fs] where
        l < len(signal) // 2.
        (3) A 1-D array of len(signal) + 1 doppler frequencies. The order of these
        frequencies is given by numpy.fft.fftfreq. Numpy's fft shift can be used
        to center these frequencies and the ambiguity matrix about the zero
        frequency for plotting.

    References:
        Najmi, A. H. "The Wigner distribution: A time-frequency analysis tool."
        Johns Hopkins APL Technical Digest 15 (1994): 298-298.
    """

    # compute autocorrelation and ambiguity
    autocov = autocovariance(signal, **kwargs)
    amb = np.fft.ifft(autocov, axis=1)

    # compute doppler shifts and lags (eta and tau respectively)
    etas = np.fft.fftfreq(len(signal), d=1/fs)
    # taus are 2X the integer shifts in ambiguity defn.
    max_shift = (amb.shape[0] - 1) // 2
    taus = 2 * np.arange(-max_shift, max_shift+1) / fs
    return amb, taus, etas


def wigner(
        signal: npt.NDArray,
        fs: float,
        detrend: bool = True,
        analytic: bool = True,
        kernel: Optional[Callable[..., npt.NDArray]] = None,
    ) -> npt.NDArray:
    """The Wigner distribution function, a representation of instantaneous
    spectral density over time and frequencies suitable for non-stationary
    signal processing.

    This transform provides the highest possible time-frequency resolved
    spectral density of a signal achievable under the Uncertainty Principle.
    This resolution comes at the cost of large cross-terms for multicomponent
    signals. These cross-terms can be mitigated by an smoothing kernel.

    Args:
        signal:
            A 1-D array real or complex signal.
        fs:
            The sampling rate in Hz at which signal was acquired.
        detrend:
            A boolean indicating if the signal should be detrended. If signal
            has a non-zero mean it should be detrended. Defaults to True.
        analytic:
            A boolean indicating if the negative frequency components should be
            removed. This prevents negative frequency aliasing in the
            time-frequency distributions. Defaults to True.
        kernel:
            A callable usually representing a low-pass filter in the ambiguity
            domain used to mask out interference terms. This extends the Wigner
            distribution function to Cohen's class of Bilinear transformations.
            For help in choosing a kernel please see phasor.kernels

    Returns:
        A 3-tuple consisting of:
        (1) A 2-D array of spectral densities over times and frequencies with
        frequencies along axis=0 and time along axis=1.
        (2) A 1-D array of frequencies from [-fs/4...fs/4]
        (3) A 1-D array of times from 0 to len(signal)/fs.

    Notes:
        Spectrograms usually run from -fs/2 to fs/2 but the WDF runs from -fs/4
        to fs/4. This reduced frequency range results from having to increment
        delays in the autocovariance by steps of two tau to have integer shifts
        of the signal. This reduces the frequency range. Please see ambiguity
        defn in Reference 1.

    References:
        Najmi, A. H. "The Wigner distribution: A time-frequency analysis tool."
        Johns Hopkins APL Technical Digest 15 (1994): 298-298.

    """

    if kernel:
        # TODO add kernel smoothed calculation
        amb, taus, etas = ambiguity(signal, fs, detrend=detrend, analytic=analytic)
        smoother = kernel(etas, taus)
        result = 2 * np.fft.fft2(smoother * amb)
    else:
        # bypass ambiguity when no smoothing kernel
        autocov = autocovariance(signal, detrend, analytic)
        result = np.fft.fft(autocov, axis=0)

    times = np.arange(0, len(signal)/fs, 1/fs)
    freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

    # place 0 frequency at center
    result = np.fft.fftshift(result, axes=0)
    freqs = np.fft.fftshift(freqs)

    return result, freqs, times


def choi_williams(etas, taus, sigma):
    """ """

    cols, rows = np.meshgrid(etas, taus)
    return np.exp(-(cols**2 * rows**2) / sigma)





if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    from phasor.data.synthetic import PACSignal

    from functools import partial


    """
    duration = 10
    fs = 32
    time = np.linspace(0, duration, duration*fs+1)
    signal = 4*np.sin(2 * np.pi * 4.5 * time) + 3*np.sin(2*np.pi*9*time) 
    signal[0:4*fs] = np.random.random(size=4*fs)
    signal[9*fs:] = np.random.random(size=(len(signal) - 9*fs))
    signal += 0.5*np.random.random(len(signal))
    plt.plot(time, signal)

    res = ambiguity(signal, fs=fs, analytic=True)

    fig, ax = plt.subplots()
    lags = np.linspace(-len(signal)/fs*2, len(signal)*fs*2, len(signal))
    print(lags)
    ax.pcolormesh(np.abs(res))
    plt.show()
    """


    """
    signal = np.array([7,4,6,3, 2])
    amb, taus, etas = ambiguity(signal, fs=100, analytic=False)
    """

    """
    #ambiguity test
    fs = 32
    duration = 12
    time = np.linspace(0, duration, duration*fs + 1)
    #signal = np.sin(2*np.pi*4.5*time)
    signal = np.sin(2 * np.pi * 4.5 * time) + np.sin(2 * np.pi * 9 * time)
    signal[:6*fs] = 0
    signal[10*fs:] = 0
    signal += np.random.random(fs*duration+1) * 0.25
    amb, taus, etas = ambiguity(signal, fs, analytic=True)

    kernel = choi_williams(etas, taus, sigma=0.01)
    
    # for plotting fftshift amb and etas
    amb = np.fft.fftshift(amb, axes=1)
    etas = np.fft.fftshift(etas)
    kern = np.fft.fftshift(kernel, axes=1)

    
    fig, axarr = plt.subplots(1,2)
    axarr[0].pcolormesh(etas, taus, np.abs(amb), shading='nearest')
    axarr[1].pcolormesh(etas, taus, kern)
    plt.show()
    """

    #wigner tests
    fs = 64
    duration = 12
    times = np.linspace(0, duration, duration*fs + 1)
    #time = np.arange(0, 5, 1/fs)
    signal = 2*np.sin(2*np.pi*4.5*times) + 2*np.sin(2*np.pi*9*times)
    #signal = np.sin(2*np.pi*4.5*times)
    signal[:6*fs] = 0
    signal[10*fs:] = 0
    signal += np.random.random(fs*duration+1) * 0.25
    t0 = time.perf_counter()
    kernel =  partial(choi_williams, sigma=1)
    w, freqs, t = wigner(signal, fs=fs, analytic=True, kernel=kernel)
    wig = np.abs(w)
    print(f'Computed Wigner in {time.perf_counter() - t0} secs')

    # FIXME need to restrict plots using nearest function to help
    fig, ax = plt.subplots()
    #ax.pcolormesh(t, freqs, np.abs(wig))
    pcm = ax.pcolormesh(t, freqs, wig, shading='nearest')
    fig.colorbar(pcm, ax=ax)
    plt.show()
    
