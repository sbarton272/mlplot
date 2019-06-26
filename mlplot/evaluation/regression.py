"""Evaluation class for regression evaluation"""
import sklearn.metrics as metrics

from . import np, sp, plt
from .evaluation import ModelEvaluation
from ..errors import InvalidArgument

class RegressionEvaluation(ModelEvaluation):
    """Single response variable regression model evaluation

    Parameters
    ----------
    y_true : list or 1d numpy array with float elements
             A collection of the true value
    y_pred : list or 1d numpy array with float elements
             A collection of the predicted value
    value_name : string
                 The name of the value/response variable being predicted
    model_name : string
                 The name of the model is used when creating plots
    """

    def __init__(self, y_true, y_pred, value_name, model_name):
        super().__init__(y_true=y_true, y_pred=y_pred, model_name=model_name)
        self.value_name = value_name

    def __repr__(self):
        return 'RegressionEvaluation(model_name={})'.format(self.model_name)

    ###################################
    # Scores

    # R2
    # Mean residual
    # Residual bias?

    ###################################
    # Plots

    def scatter(self, ax=None):
        """Plot y_true and y_pred together on a scatter plot

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        ax.scatter(self.y_true, self.y_pred)

        # Format plot
        ax.set_title('Scatter for {}'.format(self.model_name))
        ax.set_xlabel('Actual {}'.format(self.value_name))
        ax.set_ylabel('Predicted {}'.format(self.value_name))

        return ax

    def residuals(self, ax=None):
        """Plot the residuals by y_true

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        ax.scatter(self.y_true, self.y_pred - self.y_true)

        # Format plot
        ax.set_title('Residuals for {}'.format(self.model_name))
        ax.set_xlabel('Actual {}'.format(self.value_name))
        ax.set_ylabel('Residual {} (Prediction - Actual)'.format(self.value_name))

        return ax

    def residuals_histogram(self, ax=None):
        """Plot a histogram of the residuals

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        ax.hist(self.y_pred - self.y_true)

        # Format plot
        ax.set_title('Residuals Histogram for {}'.format(self.model_name))
        ax.set_xlabel('Residual {} (Prediction - Actual)'.format(self.value_name))
        ax.set_ylabel('Occurances')

        return ax
