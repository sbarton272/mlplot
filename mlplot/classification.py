"""Module containing all classification model evaluation plots"""
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from mlplot.utilities import classification_args, binarize

@classification_args
def roc_curve(y_true, y_pred, ax=None):
    """Reciever operating curve
    """
    # Compute false positive rate, true positive rate and AUC
    false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred)
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    # Create figure
    ax.plot(false_pos_rate, true_pos_rate)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Line for random
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    ax.set_title('ROC Curve with AUC {0:0.2}'.format(auc))


@classification_args
def calibration(y_true, y_pred, ax=None, n_bins='auto'):
    """Plot a calibration plot

    Calibration plots are used the determine how well the predicted values match the true value.

    This plot is as found in `sklean <http://scikit-learn.org/stable/modules/calibration.html>`_.

    Parameters
    ----------
    n_bins : int or string
             The number of bins to group y_pred. See `numpy.histogram <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html>`_
    """
    counts, bin_edges = np.histogram(y_pred, bins=n_bins, range=(0, 1))
    bin_labels = np.digitize(y_pred, bin_edges[:-1]) - 1
    fraction_positive = np.bincount(bin_labels, weights=y_true) / counts
    centers = np.bincount(bin_labels, weights=y_pred) / counts

    # Used to measure how far off from fully calibrated https://en.wikipedia.org/wiki/Brier_score
    brier = metrics.brier_score_loss(y_true=y_true, y_prob=y_pred)

    # Fraction positive
    ax.plot(centers, fraction_positive)
    ax.set_title('Calibration Brier Score {0:0.3}'.format(brier))
    ax.set_ylabel('Actual Probability')
    ax.set_xlabel('Predicted Probability')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    # Counts
    ax_r = ax.twinx()
    num_bins = len(counts)
    ax_r.bar(centers, counts, alpha=0.5, color='black', fill=True, width=0.8/num_bins)
    ylim = ax_r.get_ylim()
    ax_r.set_ylim(ylim[0], ylim[1]*10)
    ax_r.set_yticks([])
    ax_r.set_yticklabels([])

@classification_args
def precision_recall(y_true, y_pred, x_axis='recall', ax=None):
    """Plot the precision-recall curve

    An example of this plot can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.

    Parameters
    ----------
    x_axis : str 'recall' or 'threshold'
             Specify the x axis of the plot. Precision recall tends to come in 2 flavors, one precision vs recall and
             the other precion and recall vs threshold.
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_pred)

    # Create the figure
    if x_axis == 'recall':
        ax.step(recall, precision, where='post')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

    elif x_axis == 'threshold':
        ax.plot(threshold, recall[:-1], label='recall')
        ax.plot(threshold, precision[:-1], label='precision')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Precision/Recall')
        ax.legend()

    else:
        raise ValueError("x_axis can be either 'recall' or 'threshold'")

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall: Avg Precision {0:0.2f}'.format(average_precision))


@classification_args
def population_histogram(y_true, y_pred, class_labels=None, ax=None):
    """Plot histograms of the predictions grouped by class
    """
    alpha = 0.5  # Bars should be fairly transparent
    cond = y_true.astype(bool)
    ax.hist(y_pred[cond], alpha=alpha, label=class_labels[1])
    ax.hist(y_pred[~cond], alpha=alpha, label=class_labels[0])

    # Keep consistent x axis
    ax.set_xlim(0, 1)

    ax.legend()
    ax.set_title('Distribution of predictions by class')
    ax.set_ylabel('count')
    ax.set_xlabel('y_pred')


@classification_args
def confusion_matrix(y_true, y_pred, class_labels=None, threshold=0.5, ax=None):
    """Plot a heatmap for the confusion matrix

    An example of this heatmap can be found on `sklean <http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py>`_.

    Parameters
    ----------
    threshold : float
                Defines the cutoff to be considered in the asserted class
    """
    mat = metrics.confusion_matrix(y_true, binarize(y_pred, threshold))

    # Heatmap
    ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)

    # Class labels
    if class_labels:
        classes = [class_labels[0], class_labels[1]]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

    # Numbers in squares
    color_thresh = mat.max() / 2
    total = mat[:].sum()
    for (x, y), val in np.ndenumerate(mat):
        text_color = 'white' if val > color_thresh else 'black'
        text = '{0}\n({1:.1f}%)'.format(val, 100*val/total)
        ax.text(x, y, text, fontsize='large', horizontalalignment='center', color=text_color)

    ax.set_ylabel('True')
    ax.set_xlabel('Prediction')
    ax.set_title('Confusion Matrix')

@classification_args
def report_table(y_true, y_pred, class_labels=None, ax=None):
    """Generate a report table containing key stats about the dataset
    """
    tbl = []

    # Sample counts
    tbl.extend([
        ('Total count', len(y_true)),
        ('Class {} count'.format(class_labels[0]), int((y_true == 0).sum())),
        ('Class {} count'.format(class_labels[1]), int((y_true == 1).sum())),
    ])

    # Scoring
    f1 = metrics.f1_score(y_true=y_true, y_pred=binarize(y_pred))
    acc = metrics.accuracy_score(y_true=y_true, y_pred=binarize(y_pred))
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    tbl.extend([
        ('F1 Score', f1),
        ('Accuracy', acc),
        ('AUC', auc),
    ])

    # Format the rows
    rows = []
    for lbl, val in tbl:
        if isinstance(val, int):
            formatted = [lbl, val]
        else:
            formatted = [lbl, '{:.2f}'.format(val)]
        rows.append(formatted)

    ax_tbl = ax.table(cellText=rows, loc='center')
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove border
    [line.set_linewidth(0) for line in ax.spines.values()]

    # Values left justified
    cells = ax_tbl.properties()['celld']
    for row in range(len(tbl)):
        cells[row, 1]._loc = 'left'

    # Remove table borders
    for cell in cells.values():
        cell.set_linewidth(0)

    # Make cells larger
    for cell in cells.values():
        cell.set_height(0.1)

    ax.set_title('Classification Report')
