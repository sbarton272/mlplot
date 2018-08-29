"""Tests for the classification plots"""
from . import TEST_DIR, np
import mlplot.classification as clf

Y_TRUE = np.random.randint(2, size=1000)
Y_PRED = np.clip(np.random.normal(0.25, 0.3, size=Y_TRUE.shape) + Y_TRUE * 0.5, 0, 1)

def test_roc():
    """Test the ROC plot"""
    fig, ax, true_pos_rate, false_pos_rate, thresholds, auc = clf.roc(Y_TRUE, Y_PRED)
    fig.savefig(str(TEST_DIR / 'test_roc.png'))

def test_calibration():
    """Test calibration plot"""
    fig, ax, centers, fraction_positive, counts = clf.calibration(Y_TRUE, Y_PRED)
    fig.savefig(str(TEST_DIR / 'test_calibration.png'))

def test_precision_recall():
    """Test precision_recall plot"""
    fig, ax, precision, recall, average_precision = clf.precision_recall(Y_TRUE, Y_PRED)
    fig.savefig(str(TEST_DIR / 'test_precision_recall.png'))

def test_precision_recall_threshold():
    """Test precision_recall_threshold plot"""
    fig, ax, precision, recall, threshold = clf.precision_recall_threshold(Y_TRUE, Y_PRED)
    fig.savefig(str(TEST_DIR / 'test_precision_recall_threshold.png'))
