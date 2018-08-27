"""Module containing all classification model evaluation plots"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

from . import plt

# TODO validation
# TODO pass in plot
def roc(y_true, y_pred):
    """Reciever operating curve"""
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
    """Plot a calibration plot as found at http://scikit-learn.org/stable/modules/calibration.html"""
    counts, bin_edges = np.histogram(y_pred, bins=n_bins, range=(0, 1))
    bin_labels = np.digitize(y_pred, bin_edges[:-1]) - 1
    fraction_positive = np.bincount(bin_labels, weights=y_true) / counts
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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
