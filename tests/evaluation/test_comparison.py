"""Test for mlplot mutliple classification model evaluation"""
import pytest

from .. import np, output_ax

from mlplot.errors import InvalidArgument
from mlplot.evaluation import ClassificationComparison, ClassificationEvaluation
from mlplot.evaluation import RegressionComparison, RegressionEvaluation

@pytest.fixture
def cls_comp():
    """Setup an example ClassificationComparison"""
    # First model
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.3, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval1 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='train',
    )

    # Second model
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.4, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval2 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='test',
    )

    # Third model
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.4, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval3 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='validate',
    )

    return ClassificationComparison([eval1, eval2, eval3])

# TODO split apart classes and tests
@pytest.fixture
def reg_comp():
    """Setup an example RegressionComparison"""
    # First model
    y_true = np.random.normal(size=100)
    y_pred = np.random.normal(0.25, 0.3, size=y_true.shape) + y_true

    eval1 = RegressionEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        value_name='variable',
        model_name='test',
    )

    # Second model
    y_true = np.random.normal(size=100)
    y_pred = np.random.normal(0.2, 0.3, size=y_true.shape) + y_true

    eval2 = RegressionEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        value_name='variable',
        model_name='train',
    )

    # Third model
    y_true = np.random.normal(size=100)
    y_pred = np.random.normal(0.3, 0.3, size=y_true.shape) + y_true

    eval3 = RegressionEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        value_name='variable',
        model_name='validate',
    )

    return RegressionComparison([eval1, eval2, eval3])

def test_inputs():
    with pytest.raises(InvalidArgument) as exception:
        ClassificationComparison([])
    assert 'at least 2' in str(exception.value)

    with pytest.raises(InvalidArgument) as exception:
        ClassificationComparison([None, None])
    assert 'ClassificationEvaluation objects' in str(exception.value)

    # Test evals with different classes
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.3, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval1 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='train',
    )

    eval2 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['c', 'd'],
        model_name='test',
    )

    with pytest.raises(InvalidArgument) as exception:
        ClassificationComparison([eval1, eval2])
    assert 'different classes' in str(exception.value)

def test_roc_curve(cls_comp, output_ax):
    cls_comp.roc_curve(ax=output_ax)

def test_calibration(cls_comp, output_ax):
    cls_comp.calibration(ax=output_ax)

def test_precision_recall_regular(cls_comp, output_ax):
    cls_comp.precision_recall(x_axis='recall', ax=output_ax)

def test_precision_recall_threshold(cls_comp, output_ax):
    cls_comp.precision_recall(x_axis='threshold', ax=output_ax)

def test_distribution(cls_comp, output_ax):
    cls_comp.distribution(ax=output_ax)

def test_report_table(cls_comp, output_ax):
    cls_comp.report_table(ax=output_ax)