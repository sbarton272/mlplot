"""Module containing all classification model evaluation plots"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.metrics import precision_recall_curve, average_precision_score

from . import plt

# TODO validation
# TODO pass in plot
# TODO figure out return values
def roc(y_true, y_pred):
    """Reciever operating curve

    Parameters
    ----------
    y_true : np.array
    y_pred : np.array
    """
    # Compute false positive rate, true positive rate and AUC
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(false_pos_rate, true_pos_rate)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Line for random
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    ax.set_title('ROC Curve with AUC {0:0.2}'.format(auc))

    return fig, ax, true_pos_rate, false_pos_rate, thresholds, auc


def calibration(y_true, y_pred, n_bins=10):
    """Plot a calibration plot

    This plot is as found in `sklean <http://scikit-learn.org/stable/modules/calibration.html>`_
    """
    counts, bin_edges = np.histogram(y_pred, bins=n_bins, range=(0, 1))
    bin_labels = np.digitize(y_pred, bin_edges[:-1]) - 1
    fraction_positive = np.bincount(bin_labels, weights=y_true) / counts
    centers = np.bincount(bin_labels, weights=y_pred) / counts

    # Used to measure how far off from fully calibrated https://en.wikipedia.org/wiki/Brier_score
    brier = brier_score_loss(y_true=y_true, y_prob=y_pred)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))

    # Fraction positive
    axes[0].plot(centers, fraction_positive)
    axes[0].set_title('Calibration Brier Score {0:0.3}'.format(brier))
    axes[0].set_ylabel('Fraction Positive')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    # Counts
    axes[1].bar(centers, counts, width=1 / (n_bins+2), fill=False)
    axes[1].set_ylabel('Count Samples')
    axes[1].set_xlabel('Mean Bucket Prediction')

    return fig, axes, centers, fraction_positive, counts


def precision_recall(y_true, y_pred):
    """Plot the precision-recall curve

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    average_precision = average_precision_score(y_true=y_true, y_score=y_pred)

    # Create the figure
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.step(recall, precision, where='post')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('2-class Precision-Recall curve: Avg Precision {0:0.2f}'.format(average_precision))

    return fig, ax, precision, recall, average_precision


def precision_recall_threshold(y_true, y_pred):
    """Plot the precision-recall curve

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    precision, recall, threshold = precision_recall_curve(y_true=y_true, probas_pred=y_pred)

    # Create the figure
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(threshold, recall[:-1], label='recall')
    ax.plot(threshold, precision[:-1], label='precision')

    ax.set_xlabel('Threshold')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall vs Threshold')

    ax.legend()

    return fig, ax, precision, recall, threshold


def population_histogram(y_true, y_pred):
    """Plot histograms of the predictions grouped by class
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.hist(y_pred[y_true])
    ax.hist(y_pred[~y_true])

    return fig, ax
