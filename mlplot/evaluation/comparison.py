"""Plot multiple evaluation plots together for comparison"""
import matplotlib
import matplotlib.pyplot as plt

from ..errors import InvalidArgument
from .classification import ClassificationEvaluation

class ClassificationComparison():
    """TODO"""

    def __init__(self, evaluations):
        self._evaluations = evaluations

    def roc_curve(self, ax):
        ax = self._validate_axes(ax)

        # Plot each ROC curve
        for evl in self._evaluations:
            evl.roc_curve(ax=ax)

        # Remove all the extra plottings of the random line
        for i in range(1, len(2 * self._evaluations) - 1, 2):
            ax.lines[i].remove()

        # Format the plot
        ax.set_title('ROC Curve')
        ax.legend()  # Recompute legend

        return ax

    # TODO decorator
    def _validate_axes(self, ax):
        """Validate matplotlib axes or generate one if not provided"""
        if ax and not isinstance(ax, matplotlib.axes.Axes):
            raise InvalidArgument('You must pass a valid matplotlib.axes.Axes')
        elif not ax:
            _, ax = plt.subplots()
        return ax
