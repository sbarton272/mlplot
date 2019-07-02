"""Helpful decorators for plotting"""
import functools
import inspect

import matplotlib
import matplotlib.pyplot as plt

from ..errors import InvalidArgument

AX_ARG = 'ax'

def plot(plot_function):
    """Wrapper for plotting methods"""

    # Get the argument location
    argspec = inspect.getargspec(plot_function)
    arg_index = argspec.args.index(AX_ARG)

    @functools.wraps(plot_function)
    def wrapper(*args, **kwargs):
        """Call the wrapped function with an additional matplotlib axes argument"""

        # Get axes from arguments
        if AX_ARG in kwargs:
            ax = kwargs[AX_ARG]
        elif len(args) > arg_index:
            ax = args[arg_index]
        else:
            _, ax = plt.subplots()

        # Validate axis
        if not isinstance(ax, matplotlib.axes.Axes):
            raise InvalidArgument('You must pass a valid matplotlib.axes.Axes')

        # Call plot function with axes
        new_kwargs = kwargs.copy()
        new_kwargs[AX_ARG] = ax
        plot_function(*args, **new_kwargs)

        return ax

    return wrapper
