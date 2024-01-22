"""

"""

from collections import defaultdict

import numpy as np
import numpy.typing as npt

from phasor.core import numerical


def modulation_index(
        x: npt.NDArray,
        y: npt.NDArray,
        extrema: List[float, float] = [0, 360],
        nbins: int = 18,
        base=np.log10):
    """FIXME .

    Args:
        x:
            A 1-D array of inputs to be binned.
        y:
            A 1-D array of responses to signal x.
        extrema:
            The min and max value that x can take. These values need not be in x.
        nbins:
            The number of bins between extrema over which x will be digitized.

    Returns:
        A float modulation index value

    References:
        Tort et. al. Measuring Phase-Amplitude Coupling between Neuronal
        Oscillations of Different Frequencies. J. Neurophysiol. 104 1195-1210.
    """

    # estimate the density of y over binned x's and compute entropy
    density = numerical.density(x, y, extrema, nbins)
    h = numerical.shannon(density, base=base)

    # get max entropy and return Kullback-Leibler divergence
    max_entropy = base(len(density))
    return (max_entropy - h) / max_entropy





if __name__ == '__main__':

    from phasor.core.numerical import envelopes, phases, analytic
    
    amplitude = 2
    freq = 8
    fs = 500
    duration = 90
    phi = 0
    times = np.arange(0, fs *(duration + 1/fs)) / fs
    a = amplitude * np.sin(2 * np.pi * freq * times + phi)

    sa = analytic(a, axis=-1)
    ph = phases(sa)
    amp = envelopes(sa)

    pdf = densities(ph, amp)
    pdf0 = modulation_index(ph, amp)

