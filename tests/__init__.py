"""Entrypoint for tests"""
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).parent / 'output'

np.random.seed(88211293)
