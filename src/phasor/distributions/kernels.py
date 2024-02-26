"""A collection of callables for masking the doppler-lag domain of the ambiguity
function.

For multicomponent signals, the ambiguity function in the integrand of Cohen's
bilinear time-frequency distribution definition will create cross-term
interferences for real signals. Kernels that mask the doppler-lag domain of
the ambiguity function can be used to reduce the contributions of this
cross-term interference. The kernel choice will determine whether the resulting
density is real and provide different trade-offs of frequency and time
resolution. These details are explored in the documentation for each kernel
defined in this module.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt


# Protocols may have no public methods
# pylint: disable-next=too-few-public-methods
# FIXME this does not enforce a function signature it merely defines that
# kernels must be callables. This is not helpful
class Kernel(Protocol):
    """A Protocol defining the expected signature of all kernel callables."""

    def __call__(
        self,
        etas: npt.NDArray,
        taus: npt.NDArray,
        **kwargs,
    ) -> npt.NDArray:
        """All kernel callables must accept the ambiguity domain variables etas
        and taus and any required kwargs."""


def unitary(etas: npt.NDArray, taus: npt.NDArray) -> npt.NDArray:
    """Returns a kernel whose value is 1 at all eta, tau points in the ambiguity
    plane.

    This kernel is equivalent to no kernel and will result in the Wigner
    distribution function when substituted into Cohen's bilinear distribution.

    etas:
        A 1-D array of doppler frequencies.
    taus:
        A 1-D array of lags.

    Returns:
        A 2-D matrix of kernel values of shape len(taus) x len(etas)
    """

    return np.ones((len(taus), len(etas)))


def choi_williams(
    etas: npt.NDArray,
    taus: npt.NDArray,
    sigma: float,
) -> npt.NDArray:
    """The Choi-Williams kernel for cross-term interference reduction.

    The Choi-Williams kernel over the ambiguity coordinates etas and taus is
    defined as:

    kernel = np.exp(-(eta**2 * tau**2) / sigma)

    The sigma defines the dispersion of the kernel in the ambiguity plane. The
    rationale of this kernel is that in the ambiguity plane the auto-terms of
    a multicomponent signal lie near the origin and the cross terms tend away
    from the origin. This kernel acts as a low-pass filter to reduce the
    cross-term contribution.

    Args:
        etas:
            A 1-D array of doppler frequencies.
        taus:
            A 1-D array of lags.
        sigma:
            The dispersion (i.e. width) of the kernel.

    Returns:
        A 2-D kernel of shape len(taus) x len(etas).

    References:
        1. H.I. Choi and W. J. Williams, "Improved time-frequency
           representation of multicomponent signals using exponential kernels,"
           in IEEE Transactions on Acoustics, Speech, and Signal Processing,
           vol. 37, no. 6, pp. 862-871, June 1989, doi: 10.1109/ASSP.1989.28057.
    """

    cols, rows = np.meshgrid(etas, taus)
    return np.exp(-(cols**2 * rows**2) / sigma)


def rihaczek(etas: npt.NDArray, taus: npt.NDArray) -> npt.NDArray:
    """Returns the complex Rihaczek kernel over the ambiguity coordinates etas
    and taus.

    The substitution of this kernel into Cohen's definition of a bilinear
    time-frequency distribution (see phasor.distributions.bilinear) results in
    the complex Rihaczek distribution. It does not reduce cross-term
    interference but returns a complex energy distribution from which phase
    information can be extracted.

    Args:
        etas:
            A 1-D array of doppler frequencies.
        taus:
            A 1-D array of lags.

    Returns:
        A 2-D kernel of shape len(taus) x len(etas).

    References:
        1. L. Cohen, "Time-frequency distributions-a review," in Proceedings of
           the IEEE, vol. 77, no. 7, pp. 941-981, July 1989, doi:
           10.1109/5.30749.
    """

    cols, rows = np.meshgrid(etas, taus)
    return np.exp(1j * (cols * rows) / 2)
