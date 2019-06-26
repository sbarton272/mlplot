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

Work was inspired by [sklearn model evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html).

## Classification

### ROC with AUC number

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.roc_curve()
```

![ROC plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_roc_curve.png)

### Calibration

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.calibration()
```

![calibration plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_calibration.png)

### Precision-Recall

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.precision_recall(x_axis='recall')
eval.precision_recall(x_axis='thresold')
```

![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_precision_recall_regular.png)

![precision recall threshold plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_precision_recall_threshold.png)

### Distribution

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.distribution()
```

![distribution plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_distribution.png)

### Confusion Matrix

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.confusion_matrix(threshold=0.5)
```

![confusion matrix](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_confusion_matrix.png)

### Classification Report

```
from mlplot.model_evaluation import ClassifierEvaluation
eval = ClassifierEvaluation(y_true, y_pred, class_names, model_name)
eval.report_table()
```

![classification report](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_report_table.png)

## Regression

- Full report
  - Mean sqr error
  - Mean abs error
  - Target mean, std
  - R2
- Residual plot
- Scatter plot
- Histogram of regressor

## Library Cleanup

- Try in a notebook
- Logging
- Default matplotlib setup
- Multi-model comparison
- Report to generate multiple plots at once

# Development

## Publish to pypi

```
python setup.py sdist bdist_wheel
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

## Design

Basic interface thoughts
```
from mlplot.model_evaluation import ClassifierEvaluation
from mlplot.model_evaluation import RegressorEvaluation
from mlplot.model_evaluation import MultiClassifierEvaluation
from mlplot.model_evaluation import MultiRegressorEvaluation
from mlplot.model_evaluation import ModelComparison
from mlplot.feature_evaluation import *

eval = ClassifierEvaluation(y_true, y_pred)
ax = eval.roc_curve()
auc = eval.auc_score()
f1_score = eval.f1_score()
ax = eval.confusion_matrix(threshold=0.7)
```

- ModelEvaluation base class
- ClassifierEvaluation class
    - take in y_true, y_pred, class names, model_name
- RegressorEvaluation class
- MultiClassifierEvaluation class
- ModelComparison
    - takes in two evaluations of the same type

# TODO

- Formatting
- Linting
- Type checking?
- Security checks?
- Plot decorator

- Confusion
    - True class
    - Predicted class
    - Threshold
    - title Model:
- Distribution
    - count --> observations
    - y_pred --> predictions
    - legend title --> classes
    - title model:
- AUC variability plot?
