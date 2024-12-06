""" Deprecation warnings / errors.

The idea is to:
1) Add deprecation warning, but preserve functionality, so that people
   have a chance to update their code, keep it that way for a release
   >> deprecate.warn("To be deprecated, please consider using X")
2) On the next release, remove the code, and have it raise the BluePyDeprecationError,
   keep it like that for a release
   >> deprecate.fail("Deprecated, please consider using X")
3) Remove everything
"""

import warnings

from bluepy.exceptions import BluePyError


class BluePyDeprecationWarning(UserWarning):
    """ BluePy deprecation warning. """


class BluePyDeprecationError(BluePyError):
    """ BluePy deprecation error. """


def fail(msg=""):
    """ Raise a deprecation exception. """
    raise BluePyDeprecationError(msg)


def warn(msg=""):
    """ Issue a deprecation warning. """
    from bluepy import settings
    if settings.STRICT_MODE:
        fail(msg)
    else:
        warnings.warn(msg, BluePyDeprecationWarning)
