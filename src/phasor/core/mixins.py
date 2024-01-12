"""A collection of mixin classes used by phasors classes.

Mixins:
    ReprMixin:
        A mixin endowing inheritors with __repr__ and __str__ representations
        that display a developer repr and print string respectively.
"""

import inspect
from copy import deepcopy


class ReprMixin:
    """Mixin endowing inheritors with echo and print str representations.

    Displays public attributes of inheriting class, ignoring protected and
    private attributes prefixed with '_' or "__".
    """

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
