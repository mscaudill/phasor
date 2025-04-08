"""A transform for extracting the amplitude and phase components of a real
signal from the Rihaczek time-frequency distribution."""


class Rihaczek:
    """ """

    def __init__(self, distribution, freqs, time):
        """ """

        self.freqs = freqs
        self.time = time
        self.distribution = distribution

    def envelope(self, freq):
        """ """

        freq_index = np.argmin(np.abs(self.freqs - freq))
        return np.abs(self.distribution[freq_index, :])

    def phase(self, freq):
        """ """

        freq_index = np.argmin(np.abs(self.freqs - freq))
        c = self.distribution[freq_index, :]

        # return angles in [0, 2*pi)
        result = np.angle(c / np.abs(c))
        result[result < 0] += 2 * np.pi

        return result

