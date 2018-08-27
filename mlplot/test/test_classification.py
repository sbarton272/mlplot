"""Tests for the classification plots"""
import numpy as np

from . import TEST_DIR
import mlplot.classification as clf

Y_TRUE = np.random.randint(2, size=1000)
Y_PRED = np.clip(np.random.normal(0.25, 0.3, size=Y_TRUE.shape) + Y_TRUE * 0.25, 0, 1)

def test_roc():
    """Test the ROC plot"""

    fig, ax, true_pos_rate, false_pos_rate, thresholds, auc = clf.roc(Y_TRUE, Y_PRED)
    fig.savefig(str(TEST_DIR / 'test_roc.png'))
