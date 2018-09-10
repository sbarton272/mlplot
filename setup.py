"""Package setup"""
from setuptools import setup, find_packages
import sys

INSTALL_REQUIRES = [
    'matplotlib==2.2.3',
    'numpy==1.15.1',
    'scipy==1.1.0',
    'scikit-learn==0.19.2',
]
# Python 2 Dependencies
if sys.version_info[0] == 2:
    INSTALL_REQUIRES.append('pathlib==1.0.1')

setup(
    name='mlplot',
    version='0.0.0',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=INSTALL_REQUIRES,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-env'],
)
