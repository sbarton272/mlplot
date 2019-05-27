"""Test for mlplot model evaluation base class"""
import pytest

from .. import np
from mlplot.model_evaluation.model_evaluation import ModelEvaluation
from mlplot.errors import InvalidArgument

def test_inputs():
    """Test input validation"""
    # Valid data
    model_eval = ModelEvaluation(
        y_true=np.random.randint(2, size=100),
        y_pred=np.random.randint(2, size=100),
        model_name='foo',
    )

    model_eval = ModelEvaluation(
        y_true=list(range(100)),
        y_pred=list(range(100)),
        model_name='goo',
    )

    model_eval = ModelEvaluation(
        y_true=np.ones(shape=(10, 10)),
        y_pred=np.ones(shape=(10, 10)),
        model_name='zoo',
    )

    with pytest.raises(InvalidArgument):
        model_eval = ModelEvaluation(
            y_true=np.ones(shape=(10, 1)),
            y_pred=np.ones(shape=(10, 10)),
            model_name='boo',
        )