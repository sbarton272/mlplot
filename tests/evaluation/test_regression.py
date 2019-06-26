"""Test for mlplot model evaluation classification class"""
import pytest

from .. import np, output_ax
from mlplot.evaluation import RegressionEvaluation
from mlplot.errors import InvalidArgument

@pytest.fixture
def reg_eval():
    """Setup an example RegressionEvaluation"""
    y_true = np.random.normal(size=10000)
    y_pred = np.random.normal(0.25, 0.3, size=y_true.shape) + y_true

    model_eval = RegressionEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        model_name='foo',
    )
    return model_eval

def test_repr(reg_eval):
    """Check the string representation"""
    assert str(reg_eval) == 'RegressionEvaluation(model_name=foo)'
    assert repr(reg_eval) == 'RegressionEvaluation(model_name=foo)'

def test_scatter(reg_eval):
    pass

def test_residuals(reg_eval):
    pass

def test_residuals_histogram(reg_eval):
    pass
