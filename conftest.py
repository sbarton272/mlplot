"""Pytest config file"""
import os
from pathlib import Path
from subprocess import check_call

def pytest_sessionstart():
    """Before session.main() is called to remove .pyc files"""
    print('Removing .pyc files')
    check_call("find . -name '*.pyc' -delete", shell=True)

    # Specify the matplotlib test config file
    os.environ['MATPLOTLIBRC'] = str(Path(__file__).parent / 'tests')
