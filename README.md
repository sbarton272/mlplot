[![CircleCI](https://circleci.com/gh/sbarton272/mlplot.svg?style=svg)](https://circleci.com/gh/sbarton272/mlplot)
[![Documentation Status](https://readthedocs.org/projects/mlplot/badge/?version=latest)](https://mlplot.readthedocs.io/en/latest/?badge=latest)

# mlplot

Machine learning evaluation plots using [matplotlib](https://matplotlib.org/) and [sklearn](http://scikit-learn.org/). [Check out the docs.](https://mlplot.readthedocs.io/)

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
![ROC plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_roc_curve.png)

### Calibration
![calibration plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_calibration.png)

### Precision-Recall
![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_precision_recall_regular.png)

![precision recall threshold plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_precision_recall_threshold.png)

### Population Histograms
![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_population_histogram.png)

### Confusion Matrix
![confusion matrix](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_confusion_matrix.png)

![confusion matrix with threshold](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/tests.model_evaluation.test_classifier.test_confusion_matrix_with_threshold.png)

### Classification Report
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
