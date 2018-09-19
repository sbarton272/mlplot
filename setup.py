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
    long_description_content_type='text/markdown',
    url='https://mlplot.readthedocs.io/',
    author='sbarton272',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Model Evaluation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=INSTALL_REQUIRES,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
