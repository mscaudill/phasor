
import numpy as np
import scipy.signal as sps
from scipy.linalg import hankel, toeplitz

from phasor.core.arraytools import pad_axis_to

def wigner(signal, fs, analytic=True, axis=-1):
    """ """

    # I want to get this to work for ndarrays
    if analytic:
        x = sps.hilbert(signal, axis=axis)
    else:
        x = signal

    max_shift = signal.shape[axis] // 2
    
    column0 = np.pad(x[max_shift::-1], (0, max_shift))
    row0 = np.pad(x[max_shift:], (0, max_shift))
    left_shifts = toeplitz(column0, row0)
    right_shifts = np.flip(left_shifts, axis=0)

    """
    print(left_shifts, '\n')
    print(right_shifts, '\n')
    print('product ', left_shifts * right_shifts)
    """

    return np.fft.fft(np.conjugate(left_shifts) * right_shifts, axis=0)


def ambiguity(data, fs, analytic=True):
    """Returns the ambiguity of a 1-D data array.

    The ambiguity function is the short-time Fourier transform of a signal with
    a delayed copy of itself. It provides the distortion of a signal in terms of
    this delay and a doppler frequency shift. Time frequency distributions are
    built by mapping the signal to the ambiguity (delay, frequency shift) plane
    where the response is filtered before Fourier transforming back to time and
    frequency space.

    Args:
        data:
            The data whose ambiguity is to be measured.
        fs:
            The sampling rate of the data array.
        analytic:
            A boolean indicating if data should be Hilbert transformed prior to
            computing the ambiguity. This transfrom removes the negative
            frequencies to prevent aliasing. Default is True.

    Returns:
        A 3-tuple consisting of (1) a len(data) x len(data) ambiguity  matrix
        with doppler frequency shifts along the x-axis and doppler delays along
        the y-axis, (2) a 1-D array of len(data) doppler frequencies in [0,
        fs/2] and (3) a 1-D array of len(data) doppler lags in [-len(data),
        len(data)].
    Notes:
        We compute the ambiguity response by constructing Toeplitz matrices that
        hold both the forward and reverse data copies. Consider this signal [7,4,6].
        Its delay can range from -3 to +3 here are the reversed & forward delayed
        signals.

         delay  Reversed  Forward  Product
        -3      [0,0,0]   [0,0,0]  [0,0,0]
        -2      [6,0,0]   [0,0,7]  [0,0,0]
        -1      [4,6,0]   [0,7,4]  [0,42,0]
         0      [7,4,6]   [7,4,6]  [49,16,36]
         1      [0,7,4]   [4,6,0]  [0,42,0]
         2      [0,0,7]   [6,0,0]  [0,0,0]
         3      [0,0,0]   [0,0,0]  [0,0,0]

        Notice that only where delay <= len(signal) // 2 = {-1,0,1} do we have
        a non-zero product and that "reversed" is a Toeplitz and "forward" is
        a flip of reversed on 0th axis.

    References:
        Najmi, A. H. "The Wigner distribution: A time-frequency analysis tool."
        Johns Hopkins APL Technical Digest 15 (1994): 298-298.
    """

    x = np.array(data)
    if x.ndim != 1:
        msg = 'data must be exactly 1-D.'
        raise ValueError(msg)

    if analytic:
        x = sps.hilbert(x)

    # Construct Toeplitz
    max_shift = len(x) // 2 if len(x) % 2 else len(x) // 2 - 1
    first_col = pad_axis_to(x[max_shift::-1], 2 * max_shift + 1, side='right')
    first_row = pad_axis_to(x[max_shift:], len(x), side='right')
    reverse = toeplitz(first_col, first_row)
    forward = np.flip(reverse, axis=0)

    # FIXME we need to resolve the uncertainty about whether the amb is computed
    # at negative frequencies. I think it must be

    # compute ambiguity
    amb = np.fft.ifft(np.conjugate(reverse) * forward, axis=1)
    freqs = np.fft.fftfreq(len(x), d=1/fs)

    # fftshift is useful for visualizing the autoterm but need not be returned
    amb = np.fft.ifftshift(amb, axes=-1)
    freqs = np.fft.ifftshift(freqs)

    # FIXME we know b/c analytic there are no negative frequencies
    """
    amb = np.fft.ifftshift(amb, axes=-1)
    freqs = np.fft.ifftshift(freqs)
    idxs = slice(len(freqs)//2, None)
    amb = amb[:, idxs]
    freqs = freqs[idxs]
    """

    duration = len(x) / fs
    lags = 2*np.arange(-max_shift, max_shift+1) / fs
    return amb, lags, freqs



if __name__ == '__main__':

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
    ambiguity(signal, fs=100, analytic=False)
    """

    
    fs = 32
    duration = 12
    time = np.linspace(0, duration, duration*fs + 1)
    #time = np.arange(0, 5, 1/fs)
    #signal = np.sin(2*np.pi*4.5*time) + np.sin(2*np.pi*9*time)
    signal = np.sin(2*np.pi*12*time)
    signal[:6*fs] = 0
    signal[10*fs:] = 0
    #signal += np.random.random(fs*duration+1) * 0.25
    #wig = wigner(signal, fs=fs)
    amb, lags, freqs = ambiguity(signal, fs)

    fig, ax = plt.subplots()
    #ax.pcolormesh(time, freqs, np.abs(wig))
    ax.pcolormesh(freqs, lags, np.abs(amb), shading='nearest')
    #ax.pcolormesh(np.abs(amb), shading='flat')
    plt.show()

