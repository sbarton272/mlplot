"""mlplot module entrypoint"""
import logging
import os

import matplotlib.pyplot as plt

if os.environ.get('DISPLAY', '') == '':
    import matplotlib
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')
