""" BluePy configuration variables. """

import os

# All possible checks enabled / deprecated methods disallowed
STRICT_MODE = False

# Path to local VoxelBrain cache folder
# (see also: https://bbpteam.epfl.ch/documentation/voxcell-2.4.1/atlas.html#fetching-data)
ATLAS_CACHE_DIR = None


def str2bool(value):
    """ Convert environment variable value to bool. """
    if value is None:
        return False
    else:
        return value.lower() in ('y', 'yes', 'true', '1')


def load_env():
    """ Load settings from environment variables. """
    # pylint: disable=global-statement
    if 'BLUEPY_STRICT_MODE' in os.environ:
        global STRICT_MODE
        STRICT_MODE = str2bool(os.environ['BLUEPY_STRICT_MODE'])
    if 'BLUEPY_ATLAS_CACHE_DIR' in os.environ:
        global ATLAS_CACHE_DIR
        ATLAS_CACHE_DIR = os.environ['BLUEPY_ATLAS_CACHE_DIR']


load_env()
