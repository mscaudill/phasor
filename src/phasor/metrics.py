"""

"""

from collections import defaultdict

import numpy as np

from phasor.core import numerical


def modulation_index(phases, amplitudes, nbins=18):
    """ """

    # FIXME a few things to address
    # 1. This handles a single time series of phases and amplitudes, we could
    #    have this compute many at once too
    # 2. We need some simulated data like tort to test this out with
    # 3. We probably need subfunctions here that return things like the binned
    #    phase and average amplitudes. Lets consider how to break this up.
    # 4. We'll need an MI for many low and high freq bands so think about how to
    #    do this robustly -- should this be called repeatedly elsewhere or
    #    should we figure out how to handle multibands here? maybe we should
    #    take in filtered data bands ?
    # 5. Lastly there are clarity issues that we must work out

    unit = 360 if max(phases) > 2 * np.pi else 2 * np.pi

    binsize = unit//nbins
    bins = np.linspace(binsize, unit, num=nbins)
    phase_bins = np.digitize(phases, bins=bins)

    # bin the amplitudes by phase bin
    binned_amps = defaultdict(list)
    for idx, phase_bin in enumerate(phase_bins):

        binned_amps[phase_bin].append(amplitudes[idx])

    # compute the mean amplitude in each bin ensuring order
    means = [np.mean(binned_amps[b]) for b in np.unique(phase_bins)]
    # normalize to obtain discrete pseuodo PDF
    pdf = np.array(means) / np.sum(means)

    """
    # FIXME move these steps to numerical ??
    shannon = -1 * np.sum(pdf * np.log10(pdf))
    d_kl = np.log10(nbins) - shannon
    return d_kl / np.log10(nbins)
    """
    return pdf

def densities(phases, amplitudes, axis=-1, nbins=18):
    """ """

    unit = 360 if max(phases) > 2 * np.pi else 2 * np.pi

    bins = np.linspace(unit // nbins, unit, num=nbins)
    digitized = np.digitize(phases, bins=bins)

    """ Only works for 1D phases and 1D amplitudes
    means = []
    for phase_bin, _ in enumerate(bins):

        locs = np.where(digitized == phase_bin)
        means.append(np.mean(amplitudes[locs]))

    return np.array(means) / np.sum(means)
    """

    # need to determine if handling ndarrays of phases and amplitudes is really
    # necessary. It may not for cross channel MI measures which this can't
    # handle as easily as passing in each phase and amplitude series as above
    # a simple way would be to roll the channel axis of the amp array
    result = []
    x = numerical.islices(digitized, axis=axis)
    y = numerical.islices(amplitudes, axis=axis)
    for bin_series, amp_series in zip(x, y):

        means = []
        for phase_bin, _ in enumerate(bins):

            locs = np.where(bin_series == phase_bin)
            means.append(np.mean(amp_series[locs]))

        result.append(np.array(means) / np.sum(means))

    result = np.stack(result)
    shape = list(phases.shape)
    shape[axis] = nbins
    result.reshape(*shape)
    return result






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

