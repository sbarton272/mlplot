"""Module containing all classification model evaluation plots"""
from sklearn.metrics import roc_curve, roc_auc_score

from . import plt

# TODO validation
# TODO pass in plot
def roc(y_true, y_pred):
    """Reciever operating curve"""
    # Compute false positive rate, true positive rate and AUC
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)

    # Create figure
    fig, ax = plt.subplots()
    ax.plot(false_pos_rate, true_pos_rate)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Line for random
    ax.plot([0, 1], [0, 1], color='gray', linestyle='dashed')

    ax.set_title('ROC Curve with AUC {0:0.2}'.format(auc))

    return fig, ax, true_pos_rate, false_pos_rate, thresholds, auc
