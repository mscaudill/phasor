
import numpy as np
import scipy.signal as sps
from scipy.linalg import hankel, toeplitz

def ambiguity(signal, fs, analytic=True, axis=-1):
    """ """

    # I want to get this to work for ndarrays


    # FIXME there are unknown issues here as the plots are not expected for
    # ambiguity

    if analytic:
        x = sps.hilbert(signal, axis=axis)
    else:
        x = signal

    hank = hankel(x)
    print(hank)
    y = np.roll(x, shift=1, axis=0)
    lowtope = np.tril(toeplitz(y), -1)

    #FIXME I'm not sure what needs conjugating here and order!
    return np.fft.ifft(np.conjugate(hank) * lowtope, axis=-1)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from phasor.data.synthetic import PACSignal

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
    ax.pcolormesh(np.linspace(0, fs/2, duration*fs+1), time, np.abs(res))
    plt.show()
