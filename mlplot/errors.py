"""Model for library specific errors"""

class MlplotException(Exception):
    """Base error class for library. A base error class makes it easier for library consumers to filter errors from this library."""
    pass

class InvalidArgument(MlplotException):
    """Custom error for invalid function argument"""
    pass