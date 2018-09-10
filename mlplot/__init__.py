"""mlplot module entrypoint"""
import logging
import os

# Setup matplotlib based on the backend
import matplotlib

# TODO move to matplotlibrc file just for testing
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Set the visible
__all__ = []
