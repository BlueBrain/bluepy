import sys

sys.modules['libFLATIndex'] = 'Foo'

import numpy as np
import pandas as pd
import pandas.testing as pdt
from morphio import SectionType

import bluepy.index.indices as test_module
from bluepy import Cell, Section, Segment, Synapse


def test_soma_index():
    raw = np.array([
        [1, 2, 3, 4, 5],
    ], dtype=np.float64)
    pdt.assert_frame_equal(
        test_module.SomaIndex._wrap_result(raw),
        pd.DataFrame([{
            Cell.X: 1.,
            Cell.Y: 2.,
            Cell.Z: 3.,
            'radius': 4.,
            Cell.ID: 5,
        }]),
        check_like=True
    )


def test_segment_index():
    raw = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 2],
    ], dtype=np.float64)
    pdt.assert_frame_equal(
        test_module.SegmentIndex._wrap_result(raw),
        pd.DataFrame([{
            Segment.X1: 1.,
            Segment.Y1: 2.,
            Segment.Z1: 3.,
            Segment.X2: 4.,
            Segment.Y2: 5.,
            Segment.Z2: 6.,
            Segment.R1: 7.,
            Segment.R2: 8.,
            Cell.ID: 11,
            Section.ID: 22,
            Segment.ID: 33,
            Section.NEURITE_TYPE: SectionType.axon,
        }])
    )


def test_synapse_index():
    raw = np.array([
        [1, 2, 3, 4, 5, 6, 7, 11, 22, 33, 44, 1],
    ], dtype=np.float64)
    pdt.assert_frame_equal(
        test_module.SynapseIndex._wrap_result(raw),
        pd.DataFrame([{
            Synapse.PRE_X_CENTER: 1.,
            Synapse.PRE_Y_CENTER: 2.,
            Synapse.PRE_Z_CENTER: 3.,
            Synapse.POST_X_CENTER: 4.,
            Synapse.POST_Y_CENTER: 5.,
            Synapse.POST_Z_CENTER: 6.,
            Synapse.TOUCH_DISTANCE: 7.,
            'synapse_id': 11,
            'synapse_counter': 22,
            Synapse.PRE_GID: 33,
            Synapse.POST_GID: 44,
            'excitatory': True,
        }])
    )
