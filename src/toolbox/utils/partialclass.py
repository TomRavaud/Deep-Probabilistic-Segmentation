from functools import partialmethod


def partialclass(cls, *args, **kwds):
    """Create a new class by partially applying the given arguments to the
    interest class.

    Returns:
        class: A new class with the given arguments partially applied.
    """
    class Partial(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)
    return Partial

