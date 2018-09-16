"""Module containing all classification model evaluation plots"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt

from mlplot.utilities import classification_args

@classification_args
def roc(y_true, y_pred, labels=None, ax=None):
    """Reciever operating curve
    """
    # Compute false positive rate, true positive rate and AUC
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)

    # Create figure
    ax.plot(false_pos_rate, true_pos_rate)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Line for random
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    ax.set_title('ROC Curve with AUC {0:0.2}'.format(auc))


@classification_args
def calibration(y_true, y_pred, labels=None, ax=None, n_bins=10):
    """Plot a calibration plot

    Calibration plots are used the determine how well the predicted values match the true value.

    This plot is as found in `sklean <http://scikit-learn.org/stable/modules/calibration.html>`_.

    Parameters
    ----------
    n_bins : int
             The number of bins to group y_pred
    """
    counts, bin_edges = np.histogram(y_pred, bins=n_bins, range=(0, 1))
    bin_labels = np.digitize(y_pred, bin_edges[:-1]) - 1
    fraction_positive = np.bincount(bin_labels, weights=y_true) / counts
    centers = np.bincount(bin_labels, weights=y_pred) / counts

    # Used to measure how far off from fully calibrated https://en.wikipedia.org/wiki/Brier_score
    brier = brier_score_loss(y_true=y_true, y_prob=y_pred)

    # Fraction positive
    ax.plot(centers, fraction_positive)
    ax.set_title('Calibration Brier Score {0:0.3}'.format(brier))
    ax.set_ylabel('Fraction Positive')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    # Counts
    ax_r = ax.twinx()
    ax_r.bar(centers, counts, width=1 / (n_bins+2), fill=False)
    ax_r.set_ylabel('Count Samples')
    ax_r.set_xlabel('Mean Bucket Prediction')


@classification_args
def precision_recall(y_true, y_pred, labels=None, ax=None):
    """Plot the precision-recall curve

    An example of this plot can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.
    """
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    average_precision = average_precision_score(y_true=y_true, y_score=y_pred)

    # Create the figure
    ax.step(recall, precision, where='post')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('2-class Precision-Recall curve: Avg Precision {0:0.2f}'.format(average_precision))


@classification_args
def precision_recall_threshold(y_true, y_pred, labels=None, ax=None):
    """Plot the precision-recall curve

    An example of this plot can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.
    """
    precision, recall, threshold = precision_recall_curve(y_true=y_true, probas_pred=y_pred)

    # Create the figure
    ax.plot(threshold, recall[:-1], label='recall')
    ax.plot(threshold, precision[:-1], label='precision')

    ax.set_xlabel('Threshold')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall vs Threshold')

    ax.legend()


@classification_args
def population_histogram(y_true, y_pred, labels=None, ax=None):
    """Plot histograms of the predictions grouped by class
    """
    alpha = 0.5  # Bars should be fairly transparent
    cond = y_true.astype(bool)
    ax.hist(y_pred[cond], alpha=alpha)
    ax.hist(y_pred[~cond], alpha=alpha)
