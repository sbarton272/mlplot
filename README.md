[![CircleCI](https://circleci.com/gh/sbarton272/mlplot.svg?style=svg)](https://circleci.com/gh/sbarton272/mlplot)

# mlplot
Machine learning evaluation plots using matplotlib

# Plots

Starting from [sklearn](http://scikit-learn.org/stable/modules/model_evaluation.html).

## Classification

- ROC with AUC number
![ROC plot](https://raw.githubusercontent.com/sbarton272/mlplot/master/mlplot/mlplot/test/output/test_roc.png)

- Recall, precision and threshold
- Population histograms
- Residual plot --> what is the real name?
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
