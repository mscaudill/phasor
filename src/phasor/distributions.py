"""

"""

from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import scipy.signal as sps
from scipy.linalg import hankel, toeplitz

from phasor.core.arraytools import pad_axis_to


def autocorrelation(
        signal: npt.NDArray,
        detrend: bool = True,
        analytic: bool = True,
)-> npt.NDArray:
    """Returns the non-stationary autocorrelation matrix for a 1-D signal.

    The autocorrelation for a non-stationary signal is:

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
        A 2-D array of autocorrelation values with signal delay along the 0th
        axis and time along the 1st axis.

    References:
        1. Najmi, A. H. "The Wigner distribution: A time-frequency analysis
           tool." Johns Hopkins APL Technical Digest 15 (1994): 298-298.
        2. https://en.wikipedia.org/wiki/Autocorrelation

    Notes:
        We compute the autocorrelation using Toeplitz matrices that hold both
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
        (2) A 1-D array of doppler delays in [-len(signal), len(signal)]
        (3) A 1-D array of len(signal) + 1 doppler frequencies. The order of these
        frequencies is given by numpy.fft.fftfreq. Numpy's fft shift can be used
        to center these frequencies and the ambiguity matrix about the zero
        frequency for plotting.

    References:
        Najmi, A. H. "The Wigner distribution: A time-frequency analysis tool."
        Johns Hopkins APL Technical Digest 15 (1994): 298-298.
    """

    # compute autocorrelation and ambiguity
    auto_corr = autocorrelation(signal, **kwargs)
    amb = np.fft.ifft(auto_corr, axis=1)

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
        kernel: Optional[Callable[..., npt.NDArray] = None,
    ) -> npt.NDArray:
    """ """


    if kernel:
        # TODO add kernel smoothed calculation
        amb, *_ = ambiguity(signal, fs, detrend=detrend, analytic=analytic)
        result = 2 * np.fft.fft2(amb)
    else:
        auto_corr = autocorrelation(signal, detrend, analytic)
        result = np.fft.fft(auto_corr, axis=0)

    times = np.arange(0, len(signal)/fs, 1/fs)
    freqs = 1/2 * np.fft.fftfreq(result.shape[0], d=1/fs)

    # place 0 frequency at center
    result = np.fft.fftshift(result, axes=0)
    freqs = np.fft.fftshift(freqs)

    return result, freqs, times




if __name__ == '__main__':

    import time
    import matplotlib.pyplot as plt
    from phasor.data.synthetic import PACSignal

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
    #signal = np.sin(2*np.pi*4.5*time) + np.sin(2*np.pi*9*time)
    signal = np.sin(2*np.pi*12*time)
    signal[:6*fs] = 0
    signal[10*fs:] = 0
    #signal += np.random.random(fs*duration+1) * 0.25
    amb, taus, etas = ambiguity(signal, fs, analytic=False)

    # for plotting fftshift amb and etas
    amb = np.fft.ifftshift(amb, axes=1)
    etas = np.fft.ifftshift(etas)

    fig, ax = plt.subplots()
    ax.pcolormesh(etas, taus, np.abs(amb), shading='nearest')
    plt.show()
    """


    #wigner tests
    fs = 128
    duration = 12
    times = np.linspace(0, duration, duration*fs + 1)
    #time = np.arange(0, 5, 1/fs)
    signal = 2*np.sin(2*np.pi*4.5*times) + 2*np.sin(2*np.pi*18*times)
    #signal = np.sin(2*np.pi*4.5*times)
    signal[:6*fs] = 0
    signal[10*fs:] = 0
    signal += np.random.random(fs*duration+1) * 0.25
    t0 = time.perf_counter()
    w, freqs, t = wigner(signal, fs=fs, analytic=True, kernel=None)
    wig = np.abs(w)
    print(f'Computed Wigner in {time.perf_counter() - t0} secs')

    # FIXME need to restrict plots using nearest function to help
    fig, ax = plt.subplots()
    #ax.pcolormesh(t, freqs, np.abs(wig))
    ax.pcolormesh(t, freqs, wig, shading='nearest')
    plt.show()
    
