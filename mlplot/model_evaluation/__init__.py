"""mlplot.model_evaluation module entrypoint"""

import matplotlib.pyplot as plt
import numpy as np

from .classifier import ClassifierEvaluation

# Set the visible
__all__ = [
    'ClassifierEvaluation',
]
