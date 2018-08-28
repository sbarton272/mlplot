"""mlplot module entrypoint"""
import logging
import os

import matplotlib

if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from classification import roc, calibration

__all__ = [roc, calibration]
