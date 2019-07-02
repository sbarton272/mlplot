from .. import np, output_ax

from mlplot.evaluation import ClassificationComparison, ClassificationEvaluation

def test_roc_curve(output_ax):
    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.3, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval1 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='foo',
    )

    y_true = np.random.randint(2, size=10000)
    y_pred = np.clip(np.random.normal(0.25, 0.3, size=y_true.shape) + y_true * 0.5, 0, 1)

    eval2 = ClassificationEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['a', 'b'],
        model_name='bar',
    )

    comp = ClassificationComparison([eval1, eval2])

    comp.roc_curve(ax=output_ax)
