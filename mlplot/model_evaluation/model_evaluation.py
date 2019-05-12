"""Model evaluation base class"""
import matplotlib
import numpy as np

from ..errors import InvalidArgument

class ModelEvaluation():
    """Base class to provide utility functions for specific model type evaluation classes"""

    def __init__(self, y_true, y_pred, model_name):
        self.model_name = model_name

        # Convert vectors and ensure they are the same size
        self.y_true = self._to_vector(y_true, 'y_true')
        self.y_pred = self._to_vector(y_pred, 'y_pred')
        if self.y_true.shape != self.y_pred.shape:
            raise InvalidArgument('You must y_true and y_pred of the same size')

    def _validate_axes(self, ax):
        """Validate matplotlib axes or generate one if not provided"""
        if ax and not isinstance(ax, matplotlib.axes.Axes):
            raise InvalidArgument('You must pass a valid matplotlib.axes.Axes')
        else:
            _, ax = plt.subplots()
        return ax

    def _to_vector(self, collection, collection_name):
        """Convert an iterable to a 1D numpy array or raise an error if not possible"""
        try:
            return np.array(collection).ravel()
        except:
            raise InvalidArgument('Cannot convert {} to 1D numpy array'.format(collection_name))