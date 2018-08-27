"""Package setup"""
from setuptools import setup, find_packages

setup(
    name='mlplot',
    version='1.0.0',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[
        'matplotlib==2.2.3',
        'numpy==1.15.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
