"""Tests for the classification plots"""
from . import TEST_DIR
import mlplot.classification as clf

Y_TRUE = [1, 0, 1, 0, 1, 0, 1, 0]
Y_PRED = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def test_roc():
    """Test the ROC plot"""

    fig, ax, true_pos_rate, false_pos_rate, thresholds, auc = clf.roc(Y_TRUE, Y_PRED)
    fig.savefig(TEST_DIR / 'test_roc.png')
