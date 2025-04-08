"""A module for computing various phase locking features from 1-D signals."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from openseize.filtering.bases import FIR
from openseize.filtering import fir

from phasor.transforms import analytic


class PhasePower:
    """A metric of the phase to power locking between two band-limited signals."""

    def __init__(
        self,
        data: npt.NDArray[np.float_],
        fs: float,
        axis: int = -1,
        ftype: FIR = fir.Kaiser,
        transform = analytic.CompactAnalytic
    ) -> None:
        """ """

        self.data = data
        self.fs = fs
        self.axis = axis
        self.ftype = ftype
        self.analytic = transform(data, fs, axis, ftype)


    def phase(self, fpass: Sequence[float], fstop: Sequence[float], **kwargs):
        """ """

        self.phases = self.analytic.phase(fpass, fstop, **kwargs)


    def troughs(self):
        """ """

        # TODO indices are 2D rows, cols here
        self.indices = (self.phases < 0.05).nonzero()


    def powers(
        self,
        fstart=40,
        fstop=200,
        fstep=2,
        bandwidth=4,
        ):
        """ """

        segments = # an array of shape num troughs x channels x epoch
        # PREALLOCATE 
        for center in range(fstart, fstop, fstep):
            # compute filtered
            # compute envelopes
            # compute powers
            # extract and store segments



