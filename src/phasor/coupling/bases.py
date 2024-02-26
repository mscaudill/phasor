"""A collection of abstract base classes defining the interfaces for Phasor's
coupling metrics. Primarily these ABCs ensure that concrete subclasses implement
methods whose argument types and names are correct and consistent.
"""


import abc

import numpy.tying as npt


class PAC(abc.ABC):
    """This ABC declares all phase-amplitude coupling metrics to be callables
    that should accept phase and amplitude time series of NDArray type.

    Enforcement of the call signature and typing happens during static type
    checking and linting.
    """

    @abc.abstractmethod
    def __call__(
        self,
        phases: npt.NDArray,
        amplitudes: npt.NDArray,
        **kwargs,
    ) -> float:
        """Ensures all PACMetrics implement call method and allows for static
        type and signature checks."""


