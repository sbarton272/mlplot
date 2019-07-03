"""Plot multiple evaluation plots together for comparison"""
import matplotlib
import matplotlib.pyplot as plt

from ..errors import InvalidArgument
from .classification import ClassificationEvaluation
from .decorators import plot

class ClassificationComparison():
    """Compare multiple classification model evaluations"""

    def __init__(self, evaluations):
        self._evaluations = evaluations

    @plot
    def roc_curve(self, ax=None):
        ClassificationEvaluation.roc_curve.__doc__

        # Plot each ROC curve
        for evl in self._evaluations:
            evl.roc_curve(ax=ax)

        # Remove all the extra plottings of the random line
        for i in range(1, len(2 * self._evaluations) - 1, 2):
            ax.lines[i].remove()

        # Format the plot
        ax.set_title('ROC Curve')
        ax.legend()  # Recompute legend

