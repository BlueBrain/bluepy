import os
from functools import partial
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from morphio import SectionType

import bluepy.impl.connectome_nrn as test_module
from bluepy.connectome import Connectome
from bluepy.enums import Section, Segment, Synapse
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_deduce_nrn_prefix_1():
    nrn_path = os.path.join(TEST_DATA_DIR, "nrn")
    test_obj = test_module.NrnConnectome(nrn_path, None)
    assert test_obj._prefix == os.path.join(TEST_DATA_DIR, "nrn/")


def test_deduce_nrn_prefix_2():
    nrn_path = os.path.join(TEST_DATA_DIR, "nrn", "nrn.h5")
    test_obj = test_module.NrnConnectome(nrn_path, None)
    assert test_obj._prefix == os.path.join(TEST_DATA_DIR, "nrn/")


@patch('os.path.exists', return_value=True)
def test_deduce_nrn_prefix_3(_):
    nrn_path = os.path.join(TEST_DATA_DIR, "nrn", "proj_nrn.h5")
    test_obj = test_module.NrnConnectome(nrn_path, None)
    assert test_obj._prefix == os.path.join(TEST_DATA_DIR, "nrn/proj_")


def test_deduce_nrn_prefix_4():
    with pytest.raises(BluePyError):
        nrn_path = os.path.join(TEST_DATA_DIR, "nrn", "proj_nrn.h5")
        test_obj = test_module.NrnConnectome(nrn_path, None)


def test_deduce_nrn_prefix_5():
    with pytest.raises(BluePyError):
        nrn_path = os.path.join(TEST_DATA_DIR, "err")
        test_obj = test_module.NrnConnectome(nrn_path, None)


class TestNrnConnectome:
    def setup_method(self):
        circuit = Mock()
        circuit.morph.section_features.return_value = pd.DataFrame({
            Section.NEURITE_START_DISTANCE: [100., 200.],
        }, index=pd.Index([1, 2], name=Section.ID))
        circuit.morph.segment_features.return_value = pd.DataFrame({
            Segment.SECTION_START_DISTANCE: [10., 20., 30.],
        }, index=pd.MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)],
                                           names=[Section.ID, Segment.ID]))
        circuit.cells.ids = lambda x: x
        metadata = {"prop1": "string_prop", "prop2": 1}
        self.test_obj = Connectome(os.path.join(TEST_DATA_DIR, "nrn"), circuit, metadata=metadata)

    def test_metadata(self):
        assert self.test_obj.metadata == {"prop1": "string_prop", "prop2": 1}

    def test_available_properties_1(self):
        assert Synapse.NRRP not in self.test_obj.available_properties
        expected = {Synapse.PRE_GID, Synapse.POST_GID, Synapse.AXONAL_DELAY,
                    Synapse.POST_SECTION_ID,
                    Synapse.POST_SEGMENT_ID, Synapse.POST_SEGMENT_OFFSET, Synapse.PRE_SECTION_ID,
                    Synapse.PRE_SEGMENT_ID,
                    Synapse.PRE_SEGMENT_OFFSET, Synapse.G_SYNX, Synapse.U_SYN, Synapse.D_SYN,
                    Synapse.F_SYN,
                    Synapse.DTC, Synapse.TYPE, Synapse.POST_BRANCH_ORDER, Synapse.PRE_BRANCH_ORDER,
                    Synapse.POST_BRANCH_TYPE, Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR,
                    Synapse.PRE_Z_CONTOUR, Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR,
                    Synapse.POST_Z_CONTOUR, Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER,
                    Synapse.PRE_Z_CENTER, Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER,
                    Synapse.POST_Z_CENTER}

        assert self.test_obj.available_properties == expected


    def test_available_properties_2(self):
        self.test_obj._impl._version = 5
        assert Synapse.NRRP in self.test_obj.available_properties

    @patch('h5py.File.__getitem__')
    def test_version_missing_1(self, h5_mock):
        h5_mock.side_effect = KeyError
        assert self.test_obj._impl._version == 0

    @patch('h5py.File.__getitem__')
    def test_version_missing_2(self, h5_mock):
        h5_mock.side_effect = IndexError
        assert self.test_obj._impl._version == 0

    def test_synapse_properties_1(self):
        properties = [
            Synapse.PRE_GID,
            Synapse.AXONAL_DELAY,
            Synapse.POST_BRANCH_TYPE,
            Synapse.POST_X_CENTER,
        ]
        synapse_ids = [(1, 0)]
        actual = self.test_obj.synapse_properties(synapse_ids, properties)
        expected = pd.DataFrame([
            (3, 99.8944, SectionType.apical_dendrite, 69.4535),
        ],
            columns=properties,
            index=pd.MultiIndex.from_tuples(synapse_ids)
        )
        pdt.assert_frame_equal(actual[properties], expected[properties])

    def test_synapse_properties_2(self):
        expected = {
            Synapse.PRE_GID: 1,
            Synapse.POST_GID: 2,
            Synapse.AXONAL_DELAY: 52.188112,
            Synapse.D_SYN: 97.032807,
            Synapse.DTC: 7.139142,
            Synapse.F_SYN: 22.516637,
            Synapse.G_SYNX: 96.292664,
            Synapse.TYPE: 8,
            Synapse.U_SYN: 98.573523,
            Synapse.PRE_BRANCH_ORDER: 87,
            Synapse.PRE_NEURITE_DISTANCE: 282.949477,
            Synapse.PRE_SECTION_DISTANCE: 82.949477,
            Synapse.PRE_SECTION_ID: 2,
            Synapse.POST_BRANCH_ORDER: 81,
            Synapse.POST_BRANCH_TYPE: SectionType.axon,
            Synapse.POST_NEURITE_DISTANCE: 135.254529,
            Synapse.POST_SECTION_DISTANCE: 35.254529,
            Synapse.POST_SECTION_ID: 1,
            Synapse.PRE_X_CENTER: 97.296411,
            Synapse.PRE_Y_CENTER: 4.567859,
            Synapse.PRE_Z_CENTER: 92.017395,
            Synapse.PRE_X_CONTOUR: 76.997909,
            Synapse.PRE_Y_CONTOUR: 14.889615,
            Synapse.PRE_Z_CONTOUR: 78.491104,
            Synapse.POST_X_CENTER: 43.507527,
            Synapse.POST_Y_CENTER: 56.138269,
            Synapse.POST_Z_CENTER: 0.811523,
            Synapse.POST_X_CONTOUR: 76.032122,
            Synapse.POST_Y_CONTOUR: 58.542944,
            Synapse.POST_Z_CONTOUR: 37.175574,
            Synapse.TOUCH_DISTANCE: 60.112551,
        }
        result = self.test_obj.synapse_properties([(2, 1)], list(expected.keys()))
        for prop, value in expected.items():
            actual = result[prop].iloc[0]
            if isinstance(actual, np.floating):
                check = partial(npt.assert_almost_equal, decimal=5)
            else:
                check = npt.assert_equal
            check(actual, value, err_msg="{0}: {1} != {2}".format(prop, actual, value))

    def test_synapse_properties_3(self):
        properties = [Synapse.PRE_GID, Synapse.TOUCH_DISTANCE]
        pdt.assert_frame_equal(
            self.test_obj.synapse_properties([], properties),
            pd.DataFrame(columns=properties)
        )

    def test_synapse_properties_4(self):
        self.test_obj._impl._version = 4
        pytest.raises(BluePyError, self.test_obj.synapse_properties, [(2, 1)], [Synapse.NRRP])

    def test_synapse_properties_5(self):
        self.test_obj._impl._version = 5
        result = self.test_obj.synapse_properties([(2, 1)], [Synapse.NRRP]).iloc[0, 0]
        npt.assert_almost_equal(result, 99.931539, decimal=5)

    def test_synapse_properties_6(self):
        pytest.raises(BluePyError, self.test_obj.synapse_properties, [(0, 1)],
                         ['no-such-property'])

    def test_afferent_gids(self):
        npt.assert_equal(self.test_obj.afferent_gids(1), [3])
        npt.assert_equal(self.test_obj.afferent_gids(2), [1, 3])
        npt.assert_equal(self.test_obj.afferent_gids(3), [])
        npt.assert_equal(self.test_obj.afferent_gids(9999), [])

    def test_efferent_gids(self):
        npt.assert_equal(self.test_obj.efferent_gids(1), [2])
        npt.assert_equal(self.test_obj.efferent_gids(2), [])
        npt.assert_equal(self.test_obj.efferent_gids(3), [1, 2])
        npt.assert_equal(self.test_obj.efferent_gids(9999), [])

    def test_afferent_synapses(self):
        npt.assert_equal(
            self.test_obj.afferent_synapses(2, None),
            [(2, 0), (2, 1), (2, 2)]
        )

    def test_efferent_synapses(self):
        npt.assert_equal(
            self.test_obj.efferent_synapses(3, None),
            [(1, 0), (2, 2)]
        )

    def test_pair_synapses(self):
        npt.assert_equal(self.test_obj.pair_synapses(1, 3, None), [])
        npt.assert_equal(self.test_obj.pair_synapses(3, 1, None), [(1, 0)])

    def test_pathway_synapses_1(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([1, 2, 9999], [2, 3], None),
            [(2, 0), (2, 1)]
        )

    def test_pathway_synapses_2(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([2, 3], [1, 3, 9999], None),
            [(1, 0)]
        )

    def test_pathway_synapses_3(self):
        properties = [
            Synapse.PRE_GID,
            Synapse.POST_GID,
            Synapse.PRE_SEGMENT_OFFSET,
            Synapse.POST_SEGMENT_OFFSET,
            Synapse.PRE_X_CENTER,
            Synapse.POST_X_CENTER,
        ]
        actual = self.test_obj.pathway_synapses([2, 3], [1, 3, 9999], properties)
        expected = pd.DataFrame({
            Synapse.PRE_GID: 3,
            Synapse.POST_GID: 1,
            Synapse.PRE_SEGMENT_OFFSET: 31.948546,
            Synapse.POST_SEGMENT_OFFSET: 41.901894,
            Synapse.PRE_X_CENTER: 21.348898,
            Synapse.POST_X_CENTER: 69.453559,
        }, index=pd.MultiIndex.from_tuples([(1, 0)]))
        pdt.assert_frame_equal(actual, expected, check_like=True)

    def test_pathway_synapses_4(self):
        pytest.raises(BluePyError, self.test_obj.pathway_synapses, pre=None, post=None)

    def test_pathway_synapses_5(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([], [], None),
            []
        )

    def test_pathway_synapses_6(self):
        assert self.test_obj.pathway_synapses([], [], Synapse.AXONAL_DELAY).empty

    def test_iter_connections_1(self):
        it = self.test_obj._impl.iter_connections(
            [1, 3], [2], unique_gids=True, shuffle=False
        )
        assert next(it) == (1, 2, 2)
        pytest.raises(StopIteration, next, it)

    def test_iter_connections_2(self):
        it = self.test_obj._impl.iter_connections(
            [1, 3], [2], unique_gids=False, shuffle=False
        )
        assert list(it) == [(1, 2, 2), (3, 2, 1)]

    def test_iter_connections_3(self):
        it = self.test_obj._impl.iter_connections(
            [1, 3], [2], unique_gids=False, shuffle=True
        )
        assert sorted(it) == [(1, 2, 2), (3, 2, 1)]

    def test_iter_connections_4(self):
        it = self.test_obj._impl.iter_connections(
            [1, 9999], [1, 2, 3], unique_gids=False, shuffle=False
        )
        assert sorted(it) == [(1, 2, 2)]

    def test_iter_connections_5(self):
        it = self.test_obj._impl.iter_connections(
            None, [1, 1], unique_gids=True, shuffle=False
        )
        assert sorted(it) == [(3, 1, 1)]

    def test_iter_connection_6(self):
        circuit = Mock()
        circuit.cells.ids = lambda x: x
        test_obj = Connectome(os.path.join(TEST_DATA_DIR, "nrn_complete_graph"), circuit)
        it = test_obj.iter_connections([1, 2, 3], [1, 2, 3])
        assert sorted(it) == [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

        it = test_obj.iter_connections([1, 2, 3], [1, 2, 3], unique_gids=True)
        assert sorted(it) == [(1, 2), (2, 1)]

        it = test_obj.iter_connections([1, 2, 3], [1, 3], unique_gids=True)
        assert sorted(it) == [(1, 3), (2, 1)]

        it = test_obj.iter_connections([1, 3], [1, 3], unique_gids=True)
        assert sorted(it) == [(1, 3), (3, 1)]

        it = test_obj.iter_connections([1, 2, 3], [1, 3, 2], unique_gids=True)
        assert sorted(it) == [(1, 2), (2, 1)]

        it = test_obj.iter_connections([2, 3], [1, 3, 2], unique_gids=True)
        assert sorted(it) == [(2, 1), (3, 2)]

        it = test_obj.iter_connections([1, 3, 2], [2, 3], unique_gids=True)
        assert sorted(it) == [(1, 2), (2, 3)]

    def test_synapse_positions_1(self):
        actual = self.test_obj.synapse_positions([(2, 1)], 'pre', 'center')
        expected = pd.DataFrame([
            [97.296411, 4.567859, 92.017395]
        ],
            index=pd.MultiIndex.from_tuples([(2, 1)]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_synapse_positions_2(self):
        actual = self.test_obj.synapse_positions([(2, 1)], 'pre', 'contour')
        expected = pd.DataFrame([
            [76.997909, 14.889615, 78.491104]
        ],
            index=pd.MultiIndex.from_tuples([(2, 1)]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_synapse_positions_3(self):
        actual = self.test_obj.synapse_positions([(2, 1)], 'post', 'center')
        expected = pd.DataFrame([
            [43.507527, 56.138269, 0.811523]
        ],
            index=pd.MultiIndex.from_tuples([(2, 1)]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_synapse_positions_4(self):
        actual = self.test_obj.synapse_positions([(2, 1)], 'post', 'contour')
        expected = pd.DataFrame([
            [76.032122, 58.542944, 37.175574]
        ],
            index=pd.MultiIndex.from_tuples([(2, 1)]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_synapse_positions_5(self):
        pytest.raises(
            KeyError,
            self.test_obj.synapse_positions, [(2, 1)], 'err', 'center'
        )

    def test_synapse_positions_6(self):
        pytest.raises(
            KeyError,
            self.test_obj.synapse_positions, [(2, 1)], 'pre', 'err'
        )


def test_nrn_big_endian():
    nrn_path = os.path.join(TEST_DATA_DIR, "nrn-big-endian")
    test_obj = test_module.NrnConnectome(nrn_path, None)
    properties = [Synapse.AXONAL_DELAY,]
    actual_properties = test_obj.synapse_properties([(1, 0), (1, 1)], properties)
    expected_properties = pd.DataFrame(np.array([(4.391826, ), (4.354122, ), ], dtype=np.float32),
                                       columns=properties,
                                       index=pd.MultiIndex.from_tuples([(1, 0), (1, 1)])
                                       )

    pdt.assert_frame_equal(actual_properties, expected_properties)
