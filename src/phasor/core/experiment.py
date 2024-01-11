"""A module for creating Experiment objects, dataclass-like objects that store
data and metadata. These mutuable instances may be saved and are an accepted
type for all of Phasors callables.
"""

import inspect
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Self, Union

import numpy.typing as npt


class Experiment:
    """An object for storing, modifying and saving data and intermediate
    results in Phasor.

    Attrs:
        name:
            The name of this instance usually constructed from the raw data's
            filename.
        data:
            A numpy NDArray of raw EEG data to be analyzed with Phasor.
        axis:
            The sample axis of data.
        kwargs:
            Any user supplied metadata to be stored.
    """

    def __init__(
        self,
        name: str,
        data: npt.NDArray,
        axis: int = -1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize this Experiment instance."""

        self.name = name
        self.data = data
        self.axis = axis
        self.__dict__.update(kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """Pickles a dict representation of this Experiment.

        Args:
            path:
                The save location of this Experiments dict representation.

        Returns:
            None
        """

        target = Path(path).with_suffix('.pkl')
        with open(target, 'wb') as outfile:
            pickle.dump(self.asdict(), outfile)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Returns an Experiment instance from a dict representation on disk.

        Args:
            path:
                Path to file containing pickled dict representation of
                an Experiment.

        Returns:
            A new Experiment instance.
        """

        target = Path(path)
        with open(target, 'rb') as infile:
            instance = cls(**pickle.load(infile))
        return instance

    def asdict(self) -> dict[str, Any]:
        """Returns the dict representation of this Experiment."""

        return {key: value for key, value in self.__dict__.items()}

    def __repr__(self) -> str:
        """Returns string representation emulating instance construction."""

        name = type(self).__name__
        params = str(inspect.signature(type(self).__init__))
        return f'{name}{params}'

    def __str__(self) -> str:
        """Returns string representation for print call."""

        header = f"{type(self).__name__:{'-'}{'^'}{20}}"
        attrs = '\n'.join(f"{k} = {str(v)}" for k, v in self.__dict__.items())
        return '\n'.join((header, attrs))
