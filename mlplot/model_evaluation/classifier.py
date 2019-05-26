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
             A collection of the prediction probabilities.
             It is assumed that the first class in classes is the
    class_names : list with two string elements
                  These are the names of the two classes and they are used in plots.
                  The first string maps to the 0 class and the second to the 1 class.
    model_name : string
                 The name of the model is used when creating plots
    """

    def __init__(self, y_true, y_pred, class_names, model_name):
        super(y_true=y_true, y_pred=y_pred, model_name=model_name)

        # Check that y_true has only 0 and 1
        true_values = np.sort(np.unique(y_true))
        if not np.equal(true_values, np.array([0, 1])).all()
            raise InvalidArgument('')

    def _validate

    def __repr__(self):
        return '{}(0:{}, 1:{})'.format(self.model_name, self.class_names[0], self.class_names[1])
