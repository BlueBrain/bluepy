"""Implementation details utils."""

import numpy as np
from bluepy.enums import Cell, Section, Segment
from bluepy.exceptions import BluePyError

DYNAMICS_PREFIX = "@dynamics:"
IDS_DTYPE = np.int64


def add_dynamic_prefix(properties):
    """Add the dynamic prefix to a list of properties."""
    return [DYNAMICS_PREFIX + name for name in list(properties)]


def ensure_ids(a):
    """Convert a numpy array dtype into IDS_DTYPE.

    This function is here due to the https://github.com/numpy/numpy/issues/15084 numpy issue.
    It is quite unsafe to the use uint64 for the ids due to this problem where :
    numpy.uint64 + int --> float64
    numpy.uint64 += int --> float64

    This function needs to be used everywhere node_ids or edge_ids are returned.
    """
    return np.asarray(a, IDS_DTYPE)


def _inc(x):
    """Used for convertion from 0-based to 1-based ids."""
    if x is None or isinstance(x, str):
        return x
    return np.asarray(x) + 1


def _dec(x):
    """Used for convertion from 1-based to 0-based ids."""
    if x is None or isinstance(x, str):
        return x
    return np.asarray(x) - 1


def _calc_morph_features(gid_group, circuit):
    """ Morphology features for a subset of sections / segments of a given gid. """
    gid = gid_group[Cell.ID].iloc[0]
    # TODO: if `section_features` took *which* section.ids to calculate this could be sped up
    secf = circuit.morph.section_features(gid, [Section.NEURITE_START_DISTANCE])
    segf = circuit.morph.segment_features(gid, [Segment.SECTION_START_DISTANCE])
    segf = segf.join(secf)
    ret = gid_group.join(segf, on=(Section.ID, Segment.ID))

    missing = np.any(ret.isna()[segf.columns], axis=1)
    if np.any(missing):
        raise BluePyError(f"For {gid}, missing section/segment ids for the following:\n"
                          f"{ret[missing]}")

    return ret[segf.columns]
