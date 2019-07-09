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
    arg_index = None
    if AX_ARG in argspec.args:
        arg_index = argspec.args.index(AX_ARG)

    @functools.wraps(plot_function)
    def wrapper(*args, **kwargs):
        """Call the wrapped function with an additional matplotlib axes argument"""

        # Get axes from arguments
        if AX_ARG in kwargs:
            ax = kwargs[AX_ARG]
        elif arg_index and len(args) > arg_index:
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

def table(table_function):
    """Wrapper for plotting a table"""

    @plot
    @functools.wraps(table_function)
    def wrapper(*args, **kwargs):
        """Format the provided data into a matplotlib table"""
        # Get a list of tuples containing table contents
        data = table_function(*args, **kwargs)

        # Format the data rows
        rows = []
        for lbl, val in data:
            if isinstance(val, int):
                formatted = [lbl, val]
            elif isinstance(val, float):
                formatted = [lbl, '{:.2f}'.format(val)]
            else:
                formatted = [lbl, val]
            rows.append(formatted)

        # Create the table
        ax = kwargs[AX_ARG]
        table = ax.table(cellText=rows, loc='center')
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove border
        [line.set_linewidth(0) for line in ax.spines.values()]

        # Values left justified
        cells = table.properties()['celld']
        for row in range(len(data)):
            cells[row, 1]._loc = 'left'

        # Remove table borders
        for cell in cells.values():
            cell.set_linewidth(0)

        # Make cells larger
        for cell in cells.values():
            cell.set_height(0.1)

    # Decorate wrapper with plotting
    return wrapper