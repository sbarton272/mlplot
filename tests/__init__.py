"""Entrypoint for tests"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

PLOT_OUTPUT_DIR = Path(__file__).parent / 'output'

np.random.seed(88211293)

@pytest.yield_fixture
def output_ax(request):
    fig, ax = plt.subplots()
    yield ax
    fig.tight_layout()
    filename = '{}.{}.png'.format(request.function.__module__, request.function.__name__)
    fig.savefig(str(PLOT_OUTPUT_DIR / filename))
