"""Module used to parse the file locations."""
import os
from bluepy.exceptions import BluePyError


# TODO : add the paths function for circuit and Simulation here


def _find_circuit_anchor(blueconfig):
    """Find the circuit anchor for the relative paths."""
    try:
        parent = blueconfig.Run.CurrentDir
        if not os.path.isabs(parent):
            raise BluePyError("CurrentDir must be an absolute path.")
    except AttributeError:
        try:
            parent = os.path.dirname(blueconfig.path)
        except Exception:
            raise BluePyError('Trying to use relative path without defining CurrentDir in '
                              'BlueConfig AND the BlueConfig is not loaded from a file so it is '
                              'impossible to determine a parent directory.')
    return parent
