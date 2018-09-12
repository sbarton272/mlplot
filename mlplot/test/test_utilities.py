"""Test basic use of utility functions"""
import numpy as np
import pytest

from mlplot.utilities import classification_args, validate_classification_arguments

@classification_args
def simple_func(y_true, y_pred, labels, ax):
    """docstring"""

@classification_args
def docstring_func(y_true, y_pred, labels, ax, foo):
    """docstring

    Parameters
    ----------
    foo : int
          a random argument
    """
    return foo

def test_classification_wrapper():
    """Test basic classification argument wrapper"""

    # Check docstring updates
    assert 'Parameters' in simple_func.__doc__
    assert docstring_func.__doc__.count('Parameters') == 1
    assert 'foo : int' in docstring_func.__doc__

    # Check fail on invalid values
    with pytest.raises(AssertionError) as err:
        validate_classification_arguments([1], [1, 2], None)
    assert 'must be the same length' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments([1, 2, 3], [1, 2, 3], None)
    assert 'y_true should have 2 values' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b', 'a'], ['z', 'z', 'z'], None)
    assert 'y_pred must contain numeric' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b', 'a'], [1, 2, 3], None)
    assert 'y_pred must contain values between' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b'], [1, 0], {'a': 'class1', 'b': 'class2', 'c': 'class3'})
    assert 'Labels mapping should have only 2 classes' in str(err.value)

    with pytest.raises(AssertionError) as err:
        validate_classification_arguments(['a', 'b'], [1, 0], {'b': 'class2', 'c': 'class3'})
    assert 'Labels mapping should have keys' in str(err.value)

    # Success cases
    y_true, y_pred, labels = validate_classification_arguments(['a', 'b'], [1, 0], None)
    assert np.allclose(y_true, [0, 1])
    assert np.allclose(y_pred, [1, 0])
    assert labels == {0.0: 'a', 1.0: 'b'}
