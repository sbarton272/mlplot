[![CircleCI](https://circleci.com/gh/sbarton272/mlplot.svg?style=svg)](https://circleci.com/gh/sbarton272/mlplot)

# mlplot

Machine learning evaluation plots using [matplotlib](https://matplotlib.org/) and [sklearn](http://scikit-learn.org/).

## Install

```
pip install mlplot
```

ML Plot runs with python 3.5 and above! (using format strings and type annotations)

## Contributing

Create a PR!

# Plots

Work was inspired by [sklearn model evaluation](http://scikit-learn.org/stable/modules/evaluation.html).

## Classification

### ROC with AUC number

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.roc_curve()
```

https://github.com/sbarton272/mlplot/blob/master/tests/output/tests.evaluation.test_classification.test_calibration.png?raw=true
![ROC plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_roc_curve.png)

### Calibration

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.calibration()
```

![calibration plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_calibration.png)

### Precision-Recall

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.precision_recall(x_axis='recall')
eval.precision_recall(x_axis='thresold')
```

![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_precision_recall_regular.png)

![precision recall threshold plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_precision_recall_threshold.png)

### Distribution

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.distribution()
```

![distribution plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_distribution.png)

### Confusion Matrix

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.confusion_matrix(threshold=0.5)
```

![confusion matrix](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_confusion_matrix.png)

### Classification Report

```
from mlplot.evaluation import ClassificationEvaluation
eval = ClassificationEvaluation(y_true, y_pred, class_names, model_name)
eval.report_table()
```

![classification report](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_classification.test_report_table.png)

## Regression

### Scatter Plot

```
from mlplot.evaluation import RegressionEvaluation
eval = RegressionEvaluation(y_true, y_pred, class_names, model_name)
eval.scatter()
```

![scatter plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_regression.test_scatter.png)

### Residuals Plot

```
from mlplot.evaluation import RegressionEvaluation
eval = RegressionEvaluation(y_true, y_pred, class_names, model_name)
eval.residuals()
```

![scatter plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_regression.test_residuals.png)

### Residuals Histogram

```
from mlplot.evaluation import RegressionEvaluation
eval = RegressionEvaluation(y_true, y_pred, class_names, model_name)
eval.residuals_histogram()
```

![scatter plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_regression.test_residuals_histogram.png)

### Regression Report

```
from mlplot.evaluation import RegressionEvaluation
eval = RegressionEvaluation(y_true, y_pred, class_names, model_name)
eval.report_table()
```

![report table](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.evaluation.test_regression.test_report_table.png)

## Forecasts

- TBD

## Rankings

- TBD

# Development

## Publish to pypi

```
python setup.py sdist bdist_wheel
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

## Design

Basic interface thoughts
```
from mlplot.evaluation import ClassificationEvaluation
from mlplot.evaluation import RegressorEvaluation
from mlplot.evaluation import MultiClassificationEvaluation
from mlplot.evaluation import MultiRegressorEvaluation
from mlplot.evaluation import ModelComparison
from mlplot.feature_evaluation import *

eval = ClassificationEvaluation(y_true, y_pred)
ax = eval.roc_curve()
auc = eval.auc_score()
f1_score = eval.f1_score()
ax = eval.confusion_matrix(threshold=0.7)
```

- ModelEvaluation base class
- ClassificationEvaluation class
    - take in y_true, y_pred, class names, model_name
- RegressorEvaluation class
- MultiClassificationEvaluation class
- ModelComparison
    - takes in two evaluations of the same type

# TODO

- Fix distribution plot, make lines
- Add legend with R2 to regression plots
- Add tests for regression comparison
- Split apart files for comparison classes
- Add comparisons to README
