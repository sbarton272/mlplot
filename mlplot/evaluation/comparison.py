"""Plot multiple evaluation plots together for comparison"""
import matplotlib
import matplotlib.pyplot as plt

from ..errors import InvalidArgument
from . import np
from .classification import ClassificationEvaluation
from .decorators import plot, table

class ClassificationComparison():
    """Compare multiple classification model evaluations

    Parameters
    ----------
    evaluations : list of ClassificationEvaluation objects
                  A list of ClassificationEvaluation which will be plotted together
    """

    def __init__(self, evaluations):
        if len(evaluations) <= 1:
            raise InvalidArgument('Provide at least 2 evaluations to compare.')

        if any([not isinstance(evl, ClassificationEvaluation) for evl in evaluations]):
            raise InvalidArgument('Provide a list of ClassificationEvaluation objects.')

        # TODO test class names same

        self._evaluations = evaluations

    @plot
    def roc_curve(self, ax=None):
        """Plot a receiver operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        # Plot each ROC curve
        for evl in self._evaluations:
            evl.roc_curve(ax=ax)

        # Remove all the extra random line plottings (every other line)
        del ax.lines[1:-2:2]
        ax.legend()

        # Format the plot
        ax.set_title('ROC Curve')

    @plot
    def calibration(self, bins='auto', ax=None):
        """Plot a calibration plot

        Calibration plots are used the determine how well the predicted values match the true value.

        This plot is similar to `sklean <http://scikit-learn.org/stable/modules/calibration.html>`_.

        Parameters
        ----------
        n_bins : int or string
                The number of bins to group y_pred. See `numpy.histogram <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html>`_
        ax : matplotlib.axes.Axes, optional
        """
        # Generate the same bins for all plots
        y_pred = np.concatenate([evl.y_pred for evl in self._evaluations])
        _, bin_edges = np.histogram(y_pred, bins=bins, range=(0, 1))

        # Plot each ROC curve
        for evl in self._evaluations:
            evl.calibration(bins=bin_edges, ax=ax)

        # Remove all the extra random line plottings (every other line)
        del ax.lines[1:-2:2]
        ax.legend()

        # Format the plot
        ax.set_title('Calibration')

    @plot
    def precision_recall(self, x_axis='recall', ax=None):
        """Plot the precision-recall curve

        An example of this plot can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.

        Parameters
        ----------
        x_axis : str 'recall' or 'threshold'
                Specify the x axis of the plot. Precision recall tends to come in 2 flavors, one precision vs recall and
                the other precision and recall vs threshold.
        ax : matplotlib.axes.Axes, optional
        """
        # Plot each precision-recall curve
        for evl in self._evaluations:
            evl.precision_recall(x_axis=x_axis, ax=ax)

        # Format the plot
        ax.set_title('Precision Recall')

    @plot
    def distribution(self, ax=None):
        """Plot histograms of the predictions grouped by class

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        # Plot each precision-recall curve
        for evl in self._evaluations:
            evl.distribution(ax=ax)

        # Format the plot
        ax.set_title('Distribution of Predictions')

    @table
    def report_table(self, ax=None):
        """Generate a report table containing key stats about the dataset

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        # Get all table contents
        contents = []
        for evl in self._evaluations:
            tbl_ax = evl.report_table()
            table = tbl_ax.tables[0]
            cells = table.get_celld()
            text = np.array()
            for loc, cell in cells.items():
                text[loc] = cell.get_text().get_text()
            contents.append(text)
        assert False

        ax.set_title('Classification Report')

        return data
