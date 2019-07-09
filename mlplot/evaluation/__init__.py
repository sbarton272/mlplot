"""mlplot.model_evaluation module entrypoint"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from .classification import ClassificationEvaluation
from .comparison import ClassificationComparison, RegressionComparison
from .regression import RegressionEvaluation

# Set the visible classes
__all__ = [
    'ClassificationComparison',
    'ClassificationEvaluation',
    'RegressionComparison',
    'RegressionEvaluation',
]
