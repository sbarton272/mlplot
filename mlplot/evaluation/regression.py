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
    model_name : string
                 The name of the model is used when creating plots
    """

    def __init__(selfm y_true, y_pred, model_name):
        super().__init__(y_true=y_true, y_pred=y_pred, model_name=model_name)

    def __repr__(self):
        return 'RegressionEvaluation(model_name={})'.format(self.model_name)

    ###################################
    # Scores

    ###################################
    # Plots

    def scatter(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)
        return ax

    def residuals(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)
        return ax

    def residuals_histogram(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)
        return ax

    def regressor_histogram(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)
        return ax
