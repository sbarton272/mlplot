"""Evaluation class for 2-class classifier evaluation"""
import sklearn.metrics as metrics

from . import np
from .model_evaluation import ModelEvaluation
from ..errors import InvalidArgument

class ClassifierEvaluation(ModelEvaluation):
    """2-class classifier model evaluation

    Parameters
    ----------
    y_true : list or 1d numpy array with elements 0 or 1
             A collection of the true labels which can be any two values
    y_pred : list or 1d numpy array with elements 0.0 to 1.0
             A collection of the prediction probabilities of class value 1.
    class_names : list with two string elements
                  These are the names of the two classes and they are used in plots.
                  The first string maps to the 0 class and the second to the 1 class.
    model_name : string
                 The name of the model is used when creating plots
    """

    def __init__(self, y_true, y_pred, class_names, model_name):
        super().__init__(y_true=y_true, y_pred=y_pred, model_name=model_name)

        # Check y_true values
        true_values = np.sort(np.unique(self.y_true))
        if len(true_values) != 2 or not np.equal(true_values, np.array([0, 1])).all():
            raise InvalidArgument('y_true must contain only values of 0 or 1')

        # Check y_pred values
        if ((self.y_pred < 0.0) | (self.y_pred > 1.0)).any():
            raise InvalidArgument('y_pred must contain only values between [0.0, 1.0]')

        # Check class names
        self.class_names = class_names
        if len(class_names) != 2:
            raise InvalidArgument('class_names must contain two class names (strings)')

    def __repr__(self):
        return f'{self.model_name}(class_names=[{self.class_names[0]}, {self.class_names[1]}])'

    def roc_auc_score(self):
        """Return the ROC AUC score"""
        auc = metrics.roc_auc_score(y_true=self.y_true, y_score=self.y_pred)
        return auc

    def roc_curve(self, ax=None):
        """Plot a reciever operating curve

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        """
        ax = self._validate_axes(ax)

        false_pos_rate, true_pos_rate, _ = metrics.roc_curve(y_true=self.y_true, y_score=self.y_pred)

        # Create figure
        label = f'{self.model_name} AUC({self.roc_auc_score():0.2})'
        ax.plot(false_pos_rate, true_pos_rate, label=label)

        # Line for random
        ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed', label='random')

        # Styling
        ax.set_title(f'{self.model_name} ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')

        return ax
