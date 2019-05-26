"""Evaluation class for 2-class classifier evaluation"""

from .model_evaluation import ModelEvaluation
from ..errors import InvalidArgument

class ClassifierEvaluation(ModelEvaluation):
    """2-class classifier model evaluation

    # TODO doc strings
    Arguments
    ---------
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
        super(y_true=y_true, y_pred=y_pred, model_name=model_name)

        # Check y_true values
        true_values = np.sort(np.unique(y_true))
        if not np.equal(true_values, np.array([0, 1])).all()
            raise InvalidArgument('y_true must contain only values of 0 or 1')

        # Check y_pred values
        if ((a < 0.0) | (a > 1.0)).any():
            raise InvalidArgument('y_pred must contain only values between [0.0, 1.0]')

        # Check class names
        if len(class_names) != 2:
            raise InvalidArgument('class_names must contain two class names (strings)')

    def __repr__(self):
        return '{}(0:{}, 1:{})'.format(self.model_name, self.class_names[0], self.class_names[1])


