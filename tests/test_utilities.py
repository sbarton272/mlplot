"""Test basic use of utility functions"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlplot.utilities import classification_args, validate_classification_arguments

@classification_args
def simple_func(y_true, y_pred, labels, ax):
    """docstring"""

@classification_args
def docstring_func(y_true, y_pred, labels, ax, foo=None):
    """docstring

    Parameters
    ----------
    foo : int
          a random argument
    """
    return foo

def test_validate_classification_arguments():
    """Test basic classification argument validation"""

    # Check fail on invalid values
    with pytest.raises(AssertionError) as err:
        validate_classification_arguments([1], [1, 2])
    assert 'must be the same length' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments([1, 2, 3], [1, 2, 3])
    assert 'y_true should have 2 values' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b', 'a'], ['z', 'z', 'z'])
    assert 'y_pred must contain numeric' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b', 'a'], [1, 2, 3])
    assert 'y_pred must contain values between' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b'], [1, 0], {'a': 'class1', 'b': 'class2', 'c': 'class3'})
    assert 'Labels mapping should have only 2 classes' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b'], [1, 0], {'b': 'class2', 'c': 'class3'})
    assert 'Labels mapping should have keys' in str(err.value)

    # Success cases
    y_true, y_pred, labels = validate_classification_arguments(['a', 'b'], [1, 0])
    assert np.allclose(y_true, [0, 1])
    assert np.allclose(y_pred, [1, 0])
    assert labels == {0.0: 'a', 1.0: 'b'}

    y_true, y_pred, labels = validate_classification_arguments([1, 2], [1, 0])
    assert np.allclose(y_true, [0, 1])
    assert np.allclose(y_pred, [1, 0])
    assert labels == {0.0: '1', 1.0: '2'}

    y_true, y_pred, labels = validate_classification_arguments([1, 2], [1, 0], {1: 'a', 2: 'b'})
    assert np.allclose(y_true, [0, 1])
    assert np.allclose(y_pred, [1, 0])
    assert labels == {0.0: 'a', 1.0: 'b'}


def test_classification_wrapper():
    """Test basic classification argument wrapper"""

    # Check docstring updates
    assert 'Parameters' in simple_func.__doc__
    assert docstring_func.__doc__.count('Parameters') == 1
    assert 'foo : int' in docstring_func.__doc__

    # Check failure raises correct error
    with pytest.raises(ValueError) as err:
        docstring_func(['a', 'b'], [1, 0, 1], foo='bar')

    # Success case
    ax = docstring_func(['a', 'b'], [1, 0])
    assert isinstance(ax, matplotlib.axes.Axes)

    _, ax = plt.subplots()
    ax = docstring_func(['a', 'b'], [1, 0], ax=ax, foo='bar')
    assert isinstance(ax, matplotlib.axes.Axes)
