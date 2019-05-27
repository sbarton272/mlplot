"""Entrypoint for tests"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

PLOT_OUTPUT_DIR = Path(__file__).parent / 'output'

np.random.seed(88211293)

@pytest.yield_fixture
def output_ax():
    fig, ax = plt.subplots()
    yield ax
    fig.tight_layout()
    filename = str(PLOT_OUTPUT_DIR / f'{ax.get_title()}.png')
    fig.savefig(filename)
