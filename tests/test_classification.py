"""Tests for the classification plots"""
from . import TEST_DIR, np
import mlplot.classification as clf

Y_TRUE = np.random.randint(2, size=1000)
Y_PRED = np.clip(np.random.normal(0.25, 0.3, size=Y_TRUE.shape) + Y_TRUE * 0.5, 0, 1)
LABELS = {0: 'A', 1: 'B'}

def save_fig(ax, name):
    """Helper to save the plot"""
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(str(TEST_DIR / '{}.png'.format(name)))

def test_roc():
    """Test the ROC plot"""
    ax = clf.roc(Y_TRUE, Y_PRED)
    save_fig(ax, 'test_roc')

def test_calibration():
    """Test calibration plot"""
    ax = clf.calibration(Y_TRUE, Y_PRED)
    save_fig(ax, 'test_calibration')

def test_precision_recall():
    """Test precision_recall plot"""
    ax = clf.precision_recall(Y_TRUE, Y_PRED)
    save_fig(ax, 'test_precision_recall')

def test_precision_recall_threshold():
    """Test precision_recall_threshold plot"""
    ax = clf.precision_recall_threshold(Y_TRUE, Y_PRED)
    save_fig(ax, 'test_precision_recall_threshold')

def test_population_histogram():
    """Test population_histogram plot"""
    ax = clf.population_histogram(Y_TRUE, Y_PRED, LABELS)
    save_fig(ax, 'test_population_histogram')

    ax = clf.population_histogram(Y_TRUE, Y_PRED)
    save_fig(ax, 'test_population_histogram_1')
