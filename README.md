[![CircleCI](https://circleci.com/gh/sbarton272/mlplot.svg?style=svg)](https://circleci.com/gh/sbarton272/mlplot)
[![Documentation Status](https://readthedocs.org/projects/mlplot/badge/?version=latest)](https://mlplot.readthedocs.io/en/latest/?badge=latest)

# mlplot

Machine learning evaluation plots using [matplotlib](https://matplotlib.org/) and [sklearn](http://scikit-learn.org/). [Check out the docs.](https://mlplot.readthedocs.io/)

## Install

```
pip install mlplot
```

ML Plot is runs with python 2.7, 3.5 and 3.6!

## Contributing

Create a PR!

# Plots

Starting from [sklearn](http://scikit-learn.org/stable/modules/model_evaluation.html).

## Classification

### ROC with AUC number
![ROC plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_roc.png)

### Calibration
![calibration plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_calibration.png)

### Precision-Recall
![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_precision_recall.png)
![precision recall threshold plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_precision_recall_threshold.png)

### Population Histograms
![precision recall curve plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_population_histogram.png)

### Confusion Matrix
![confusion matrix](https://raw.githubusercontent.com/sbarton272/mlplot/master/tests/output/test_confusion_matrix.png)

### TODO
- Confusion matrix
- Full report
  - Accuracy score
  - F1 score
  - Number of samples of each class

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
