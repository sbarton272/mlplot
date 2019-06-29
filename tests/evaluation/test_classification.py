"""Test for mlplot model evaluation classification class"""
import pytest

from .. import np, output_ax
from mlplot.evaluation import ClassificationEvaluation
from mlplot.errors import InvalidArgument

@pytest.fixture
def clf_eval():
    """Setup an example ClassificationEvaluation"""
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.3, size=y_true.shape) + y_true * 0.5, 0, 1)

    model_eval = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='foo',
    )
    return model_eval

def test_inputs():
    """Check init argument validation"""
    with pytest.raises(InvalidArgument) as exception:
        ClassificationEvaluation(
            y_true=[0, 1, 2],
            y_pred=[0.2, 0.3, 0.1],
            class_names=['a', 'b'],
            model_name='foo',
        )
    assert 'y_true' in str(exception.value)

    with pytest.raises(InvalidArgument) as exception:
        ClassificationEvaluation(
            y_true=[0, 2],
            y_pred=[0.2, 0.3, 0.1],
            class_names=['a', 'b'],
            model_name='foo',
        )
    assert 'y_true' in str(exception.value)

    with pytest.raises(InvalidArgument) as exception:
        ClassificationEvaluation(
            y_true=[0, 1, 1],
            y_pred=[-0.2, 0.3, 1.1],
            class_names=['a', 'b'],
            model_name='foo',
        )
    assert 'y_pred' in str(exception.value)

    with pytest.raises(InvalidArgument) as exception:
        ClassificationEvaluation(
            y_true=[0, 1, 1],
            y_pred=[0.2, 0.3, 0.1],
            class_names=[1, 2],
            model_name='foo',
        )
    assert 'class_names' in str(exception.value)

    with pytest.raises(InvalidArgument) as exception:
        ClassificationEvaluation(
            y_true=[0, 1, 1],
            y_pred=[0.2, 0.3, 0.1],
            class_names=['a'],
            model_name='foo',
        )
    assert 'class_names' in str(exception.value)

def test_repr(clf_eval):
    """Check the string representation"""
    assert str(clf_eval) == 'ClassificationEvaluation(model_name=foo, class_names=[a, b])'
    assert repr(clf_eval) == 'ClassificationEvaluation(model_name=foo, class_names=[a, b])'

def test_roc_auc_score(clf_eval):
    assert round(clf_eval.roc_auc_score(), 2) == 0.88

def test_average_precision_score(clf_eval):
    assert round(clf_eval.average_precision_score(), 2) == 0.88

def test_f1_score(clf_eval):
    assert round(clf_eval.f1_score(), 2) == 0.88

def test_accuracy_score(clf_eval):
    assert round(clf_eval.accuracy_score(), 2) == 0.88

def test_roc_curve(clf_eval, output_ax):
    clf_eval.roc_curve(ax=output_ax)

def test_calibration(clf_eval, output_ax):
    clf_eval.calibration(ax=output_ax)

def test_precision_recall_regular(clf_eval, output_ax):
    clf_eval.precision_recall(x_axis='recall', ax=output_ax)

def test_precision_recall_threshold(clf_eval, output_ax):
    clf_eval.precision_recall(x_axis='threshold', ax=output_ax)

def test_distribution(clf_eval, output_ax):
    clf_eval.distribution(ax=output_ax)

def test_confusion_matrix(clf_eval, output_ax):
    clf_eval.confusion_matrix(threshold=0.4, ax=output_ax)

def test_report_table(clf_eval, output_ax):
    clf_eval.report_table(ax=output_ax)
