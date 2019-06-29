"""Model evaluation base class"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..errors import InvalidArgument

class ModelEvaluation():
    """Base class to provide utility functions for specific model type evaluation classes"""

    def __init__(self, y_true, y_pred, model_name):
        # Convert vectors and ensure they are the same size
        self.y_true = self._to_vector(y_true, 'y_true')
        self.y_pred = self._to_vector(y_pred, 'y_pred')
        if self.y_true.shape != self.y_pred.shape:
            raise InvalidArgument('You must have y_true and y_pred of the same size')

        # No nan values are allowed
        if np.isnan(self.y_true).any() or np.isnan(self.y_pred).any():
            raise InvalidArgument('No nan values allowed in y_true or y_pred')

        self.model_name = model_name

    def _to_vector(self, collection, collection_name):
        """Convert an iterable to a 1D numpy array or raise an error if not possible"""
        try:
            return np.array(collection).ravel()
        except:
            raise InvalidArgument('Cannot convert {} to 1D numpy array'.format(collection_name))

    def _validate_axes(self, ax):
        """Validate matplotlib axes or generate one if not provided"""
        if ax and not isinstance(ax, matplotlib.axes.Axes):
            raise InvalidArgument('You must pass a valid matplotlib.axes.Axes')
        elif not ax:
            _, ax = plt.subplots()
        return ax

    def _format_table(self, table, ax):
        """Matplotlib tables don't behave well by default"""
        # Format the rows
        rows = []
        for lbl, val in tbl:
            if isinstance(val, int):
                formatted = [lbl, val]
            else:
                formatted = [lbl, '{:.2f}'.format(val)]
            rows.append(formatted)

        ax_tbl = ax.table(cellText=rows, loc='center')
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove border
        [line.set_linewidth(0) for line in ax.spines.values()]

        # Values left justified
        cells = ax_tbl.properties()['celld']
        for row in range(len(tbl)):
            cells[row, 1]._loc = 'left'

        # Remove table borders
        for cell in cells.values():
            cell.set_linewidth(0)

        # Make cells larger
        for cell in cells.values():
            cell.set_height(0.1)
