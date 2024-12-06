""" API Classes for interactive use of BluePy."""
from bluepy.utils import deprecate

deprecate.warn("The v1 module as been removed. 'from bluepy import' now uses "
               "directly the v2 implementation of the objects.")
