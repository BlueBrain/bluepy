"""Exceptions used throughout package"""


class BluePyError(Exception):
    """base bluepy exception"""


def require(cond, msg=""):
    """ Raise BluePyError if the `cond` is not met. """
    if not cond:
        raise BluePyError(msg)
