""" Utilities for interacting with the spatial indexer. """
import pandas as pd
from morphio import SectionType

from bluepy.index.dias import DIASIndex

from bluepy.enums import Cell, Section, Segment, Synapse


class SomaIndex(DIASIndex):
    """ SOMA DIASIndex wrapper. """
    COLUMNS = [
        Cell.X,
        Cell.Y,
        Cell.Z,
        'radius',
        Cell.ID,
    ]

    @classmethod
    def _wrap_result(cls, result):
        result = pd.DataFrame(result, columns=cls.COLUMNS)
        result[Cell.ID] = result[Cell.ID].astype(int)
        return result


def _map_section_type(x):
    """ Map brion::SectionType enum to neurom.NeuriteType. """
    return {
        1: SectionType.soma,
        2: SectionType.axon,
        3: SectionType.basal_dendrite,
        4: SectionType.apical_dendrite,
    }[x]


class SegmentIndex(DIASIndex):
    """ SEGMENT DIASIndex wrapper. """
    COLUMNS = [
        Segment.X1,
        Segment.Y1,
        Segment.Z1,
        Segment.X2,
        Segment.Y2,
        Segment.Z2,
        Segment.R1,
        Segment.R2,
        Cell.ID,
        Section.ID,
        Segment.ID,
        Section.NEURITE_TYPE,
    ]

    @classmethod
    def _wrap_result(cls, result):
        result = pd.DataFrame(result, columns=cls.COLUMNS)
        for column in ['gid', Section.ID, Segment.ID]:
            result[column] = result[column].astype(int)
        result[Section.NEURITE_TYPE] = result[Section.NEURITE_TYPE].apply(_map_section_type)
        return result


class SynapseIndex(DIASIndex):
    """ SYNAPSE DIASIndex wrapper. """
    COLUMNS = [
        Synapse.PRE_X_CENTER,
        Synapse.PRE_Y_CENTER,
        Synapse.PRE_Z_CENTER,
        Synapse.POST_X_CENTER,
        Synapse.POST_Y_CENTER,
        Synapse.POST_Z_CENTER,
        Synapse.TOUCH_DISTANCE,
        'synapse_id',
        'synapse_counter',
        Synapse.PRE_GID,
        Synapse.POST_GID,
        'excitatory',
    ]

    @classmethod
    def _wrap_result(cls, result):
        result = pd.DataFrame(result, columns=cls.COLUMNS)
        for column in ['synapse_id', 'synapse_counter', Synapse.PRE_GID, Synapse.POST_GID]:
            result[column] = result[column].astype(int)
        for column in ['excitatory']:
            result[column] = result[column].astype(bool)
        return result
