import os
import sys
from functools import partial
from unittest.mock import Mock, patch

import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from morphio import SectionType
from utils import copy_file

import bluepy.impl.connectome_sonata as test_module
from bluepy.cells import CellCollection
from bluepy.connectome import Connectome
from bluepy.enums import Section, Segment, Synapse
from bluepy.exceptions import BluePyError
from bluepy.utils.deprecate import BluePyDeprecationError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def _check_series(actual, expected):
    pdt.assert_series_equal(actual, expected, check_dtype=False, check_index_type=False)


def _check_dataframe(actual, expected):
    pdt.assert_frame_equal(actual, expected, check_dtype=False, check_index_type=False)


def test_estimate_range_size_1():
    func = lambda x: Mock(ranges=np.zeros(x))
    actual = test_module._estimate_range_size(func, [10, 20, 30], n=5)
    npt.assert_equal(actual, 20)


def test_estimate_range_size_2():
    func = lambda x: Mock(ranges=[42])
    actual = test_module._estimate_range_size(func, range(10))
    npt.assert_equal(actual, 1)


def test_estimate_range_size_3():
    func = None
    pytest.raises(AssertionError, test_module._estimate_range_size, func, [])


def test__resolve_sonata_path_1():
    path = os.path.join(TEST_DATA_DIR, "edges_old.sonata")
    assert test_module._resolve_sonata_path(path) == path


def test__resolve_sonata_path_2():
    assert test_module._resolve_sonata_path(TEST_DATA_DIR) == os.path.join(TEST_DATA_DIR, "edges.sonata")


def test__resolve_sonata_path_3():
    with patch('os.path.exists', side_effect=[False, True]):
        assert test_module._resolve_sonata_path(TEST_DATA_DIR) ==  os.path.join(TEST_DATA_DIR, "edges.h5")


def test__resolve_sonata_path_4():
    with patch('os.path.exists', side_effect=[False, False]):
        pytest.raises(BluePyError, test_module._resolve_sonata_path, TEST_DATA_DIR)


def test__check_sonata_file():
    default_file = os.path.join(TEST_DATA_DIR, "edges.sonata")
    for dataset in ['source_node_id', 'target_node_id']:
        with copy_file(default_file) as filepath:
            # still have the .attrs["node_population"] attribute. It should not raise.
            test_module._check_sonata_file(filepath)
            with h5py.File(filepath, "r+") as h5:
                del h5[f"edges/default/{dataset}"].attrs["node_population"]
            # the .attrs["node_population"] attribute is removed. It should raise.
            with pytest.raises(BluePyError):
                test_module._check_sonata_file(filepath)


class TestSonataConnectomeMvdCells:
    def setup_method(self):
        self.circuit = Mock()
        self.circuit.morph.section_features.return_value = pd.DataFrame({
            Section.NEURITE_START_DISTANCE: [100., 200.],
        }, index=pd.Index([1, 2], name=Section.ID))
        self.circuit.morph.segment_features.return_value = pd.DataFrame({
            Segment.SECTION_START_DISTANCE: [10., 20., 30.],
        }, index=pd.MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)],
                                           names=[Section.ID, Segment.ID]))
        self.circuit.cells = CellCollection(os.path.join(TEST_DATA_DIR, "circuit.mvd3"))
        metadata = {"prop1": "string_prop", "prop2": 1}
        self.test_obj = Connectome(os.path.join(TEST_DATA_DIR, "edges.sonata"),
                                   self.circuit, metadata=metadata)

    def test_multiple_pop_fails(self):
        sonata_path = os.path.join(TEST_DATA_DIR, "edges.sonata")
        with patch('libsonata.EdgeStorage.population_names',
                   return_values=["1", "2"]):
            with pytest.raises(BluePyError):
                test_module.SonataConnectome(sonata_path, None)

    def test_metadata(self):
        assert self.test_obj.metadata == {"prop1": "string_prop", "prop2": 1}

    def test_available_properties(self):
        expected = {Synapse.AXONAL_DELAY, Synapse.D_SYN, Synapse.F_SYN, Synapse.DTC, Synapse.G_SYNX,
                    Synapse.U_SYN, Synapse.TYPE, Synapse.POST_SECTION_ID, Synapse.POST_SEGMENT_ID,
                    Synapse.POST_SEGMENT_OFFSET, Synapse.PRE_SECTION_ID,
                    Synapse.PRE_SEGMENT_ID, Synapse.PRE_SEGMENT_OFFSET,
                    Synapse.POST_BRANCH_TYPE, Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER,
                    Synapse.POST_Z_CENTER, Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR,
                    Synapse.POST_Z_CONTOUR, Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER,
                    Synapse.PRE_Z_CENTER, Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR,
                    Synapse.PRE_Z_CONTOUR, Synapse.POST_GID, Synapse.PRE_GID,
                    Synapse.CONDUCTANCE_RATIO,
                    Synapse.U_HILL_COEFFICIENT, 'absolute_efficacy', '@dynamics:param1',
                    Synapse.POST_SECTION_DISTANCE, Synapse.POST_NEURITE_DISTANCE,
                    Synapse.PRE_SECTION_DISTANCE, Synapse.PRE_NEURITE_DISTANCE,
                    Synapse.TOUCH_DISTANCE
                    }
        assert self.test_obj.available_properties == expected

    def test_available_properties_no_dynamics(self):
        # edges.h5 does not have the PRE/POST segment Id and off set and the
        # new Synapse.CONDUCTANCE_RATIO, Synapse.U_HILL_COEFFICIENT and does not have a
        # dynamics_params group in group 0
        # Should not have at the end the DISTANCES
        test_obj = Connectome(os.path.join(TEST_DATA_DIR, "edges.h5"), {}, metadata={})
        expected = {Synapse.AXONAL_DELAY, Synapse.D_SYN, Synapse.F_SYN, Synapse.DTC, Synapse.G_SYNX,
                    Synapse.U_SYN, Synapse.TYPE, Synapse.POST_SECTION_ID, Synapse.PRE_SECTION_ID,
                    Synapse.POST_BRANCH_TYPE, Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER,
                    Synapse.POST_Z_CENTER, Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR,
                    Synapse.POST_Z_CONTOUR, Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER,
                    Synapse.PRE_Z_CENTER, Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR,
                    Synapse.PRE_Z_CONTOUR, Synapse.POST_GID, Synapse.PRE_GID, 'absolute_efficacy',
                    Synapse.TOUCH_DISTANCE}
        assert test_obj.available_properties == expected

    def test_synapse_properties_1(self):
        properties = [
            Synapse.PRE_GID,
            Synapse.AXONAL_DELAY,
            Synapse.POST_BRANCH_TYPE,
        ]
        synapse_ids = [0, 1]
        actual = self.test_obj.synapse_properties(synapse_ids, properties)
        expected = pd.DataFrame([
            (3, 99.8945, SectionType.soma),
            (1, 88.1862, SectionType.basal_dendrite),
        ],
            columns=properties,
            index=synapse_ids
        )
        _check_dataframe(actual, expected)

    def test_synapse_properties_2(self):
        expected = {
            Synapse.PRE_GID: 1,
            Synapse.POST_GID: 2,
            Synapse.AXONAL_DELAY: 52.188112,
            Synapse.D_SYN: 97,
            Synapse.DTC: 7,
            Synapse.F_SYN: 22,
            Synapse.G_SYNX: 96.292664,
            Synapse.TYPE: 8,
            Synapse.U_SYN: 98.573523,
            # Synapse.PRE_BRANCH_ORDER: 87, # not in "new" sonata files
            Synapse.PRE_NEURITE_DISTANCE: 282.949477,
            Synapse.PRE_SECTION_DISTANCE: 82.949477,
            Synapse.PRE_SECTION_ID: 2,
            # Synapse.POST_BRANCH_ORDER: 81, # not in "new" sonata files
            Synapse.POST_BRANCH_TYPE: SectionType.soma,
            Synapse.POST_NEURITE_DISTANCE: 135.254529,
            Synapse.POST_SECTION_DISTANCE: 35.254529,
            Synapse.POST_SECTION_ID: 1,
            Synapse.PRE_X_CENTER: 97.296411,
            Synapse.TOUCH_DISTANCE: 60.112551,
            Synapse.CONDUCTANCE_RATIO: 0.75,
            Synapse.U_HILL_COEFFICIENT: 1.94,
        }
        result = self.test_obj.synapse_properties([2], expected.keys())
        for prop, value in expected.items():
            actual = result[prop].iloc[0]
            if isinstance(actual, np.floating):
                check = partial(npt.assert_almost_equal, decimal=5)
            else:
                check = npt.assert_equal
            check(actual, value, err_msg="{0}: {1} != {2}".format(prop, actual, value))

    def test_synapse_properties_3(self):
        properties = [Synapse.PRE_GID, Synapse.TOUCH_DISTANCE]
        _check_dataframe(
            self.test_obj.synapse_properties([], properties),
            pd.DataFrame(columns=properties)
        )

    def test_synapse_properties_4(self):
        pytest.raises(
            BluePyError,
            self.test_obj.synapse_properties, [0], ['err']
        )

    def test_synapse_properties_not_in_enums(self):
        synapse_ids = [0, 1, 2]
        properties = ['absolute_efficacy']
        expected = pd.DataFrame([82, 43, 99], columns=properties, index=synapse_ids)
        _check_dataframe(self.test_obj.synapse_properties(synapse_ids, properties=properties),
                         expected)

    def test_synapse_properties_not_using_enums(self):
        synapse_ids = [0, 1, 2]
        properties = ["facilitation_time"]  # == SYNAPSE_PROPERTIES[Synapse.F_SYN]
        expected = pd.DataFrame([74, 94, 22], columns=properties, index=synapse_ids)
        _check_dataframe(self.test_obj.synapse_properties(synapse_ids, properties=properties),
                         expected)

    def test_synapse_properties_dynamics(self):
        synapse_ids = [0, 1, 2]
        properties = ["@dynamics:param1"]
        expected = pd.DataFrame([0.0, 1.0, 2.0], columns=properties, index=synapse_ids)
        _check_dataframe(self.test_obj.synapse_properties(synapse_ids, properties=properties),
                         expected)

    def test_synapse_properties_touch_distance_fail(self):
        with patch('bluepy.impl.connectome_sonata.SonataConnectome.available_properties',
                   return_values={}):
            with pytest.raises(BluePyError):
                self.test_obj.synapse_properties([0], Synapse.TOUCH_DISTANCE)

    def test_synapse_positions_1(self):
        actual = self.test_obj.synapse_positions([2], 'pre', 'center')
        expected = pd.DataFrame([
            [97.296411, 4.567859, 92.017395]
        ],
            index=[2],
            columns=['x', 'y', 'z']
        )
        _check_dataframe(actual, expected)

    def test_synapse_positions_2(self):
        actual = self.test_obj.synapse_positions([2], 'pre', 'contour')
        expected = pd.DataFrame([
            [76.997909, 14.889615, 78.491104]
        ],
            index=[2],
            columns=['x', 'y', 'z']
        )
        _check_dataframe(actual, expected)

    def test_synapse_positions_3(self):
        actual = self.test_obj.synapse_positions([2], 'post', 'center')
        expected = pd.DataFrame([
            [43.507527, 56.138269, 0.811523]
        ],
            index=[2],
            columns=['x', 'y', 'z']
        )
        _check_dataframe(actual, expected)

    def test_synapse_positions_4(self):
        actual = self.test_obj.synapse_positions([2], 'post', 'contour')
        expected = pd.DataFrame([
            [76.032122, 58.542944, 37.175574]
        ],
            index=[2],
            columns=['x', 'y', 'z']
        )
        _check_dataframe(actual, expected)

    def test_synapse_positions_5(self):
        pytest.raises(
            KeyError,
            self.test_obj.synapse_positions, [2], 'err', 'center'
        )

    def test_synapse_positions_6(self):
        pytest.raises(
            KeyError,
            self.test_obj.synapse_positions, [2], 'pre', 'err'
        )

    def test_synapse_properties_5(self):
        prop = Synapse.AXONAL_DELAY
        actual = self.test_obj.synapse_properties([2], prop)
        expected = pd.Series([52.188112], index=[2], name=prop)
        _check_series(actual, expected)

    def test_afferent_gids(self):
        npt.assert_equal(self.test_obj.afferent_gids(1), [3])
        npt.assert_equal(self.test_obj.afferent_gids(2), [1, 3])
        npt.assert_equal(self.test_obj.afferent_gids(3), [])
        # fix for sonata==0.0.1 out-of-range gids returns empty list
        npt.assert_equal(self.test_obj.afferent_gids(9999), [])
        npt.assert_equal(self.test_obj.afferent_gids(-1), [])
        # some problems have been seen with the specific -1 value so testing an other one
        npt.assert_equal(self.test_obj.afferent_gids(-15), [])
        # pytest.raises(Exception, self.test_obj.afferent_gids, 9999)  # TODO: replace with SonataError

    def test_efferent_gids(self):
        npt.assert_equal(self.test_obj.efferent_gids(1), [2])
        npt.assert_equal(self.test_obj.efferent_gids(2), [])
        npt.assert_equal(self.test_obj.efferent_gids(3), [1, 2])
        # fix for sonata==0.0.1 out-of-range gids returns empty list
        npt.assert_equal(self.test_obj.efferent_gids(9999), [])
        npt.assert_equal(self.test_obj.efferent_gids(-1), [])
        # some problems have been seen with the specific -1 value so testing an other one
        npt.assert_equal(self.test_obj.efferent_gids(-15), [])
        # pytest.raises(Exception, self.test_obj.efferent_gids, 9999)  # TODO: replace with SonataError

    def test_afferent_synapses_1(self):
        npt.assert_equal(
            self.test_obj.afferent_synapses(2, None),
            [1, 2, 3]
        )

    def test_afferent_synapses_2(self):
        properties = [Synapse.AXONAL_DELAY]
        _check_dataframe(
            self.test_obj.afferent_synapses(2, properties),
            pd.DataFrame([
                [88.1862],
                [52.1881],
                [11.1058],
            ],
                columns=properties, index=[1, 2, 3]
            )
        )

    def test_efferent_synapses_1(self):
        npt.assert_equal(
            self.test_obj.efferent_synapses(3, None),
            [0, 3]
        )

    def test_efferent_synapses_2(self):
        properties = [Synapse.AXONAL_DELAY]
        _check_dataframe(
            self.test_obj.efferent_synapses(3, properties),
            pd.DataFrame([
                [99.8945],
                [11.1058],
            ],
                columns=properties, index=[0, 3]
            )
        )

    def test_pair_synapses_1(self):
        npt.assert_equal(self.test_obj.pair_synapses(1, 3, None), [])

    def test_pair_synapses_2(self):
        actual = self.test_obj.pair_synapses(1, 3, [Synapse.AXONAL_DELAY])
        assert actual.empty

    def test_pair_synapses_3(self):
        assert self.test_obj.pair_synapses(3, 1, None) == [0]

    def test_pair_synapses_4(self):
        properties = [Synapse.AXONAL_DELAY]
        _check_dataframe(
            self.test_obj.pair_synapses(3, 1, properties),
            pd.DataFrame([
                [99.8945],
            ],
                columns=properties, index=[0]
            )
        )

    def test_pathway_synapses_1(self):
        properties = [Synapse.AXONAL_DELAY]
        _check_dataframe(
            self.test_obj.pathway_synapses([1, 2], [2, 3], properties),
            pd.DataFrame([
                [88.1862],
                [52.1881],
            ],
                columns=properties, index=[1, 2]
            )
        )

    def test_pathway_synapses_2(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([2, 3], [1, 3], None),
            [0]
        )

    def test_pathway_synapses_3(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([1, 2], None, None),
            [1, 2]
        )

    def test_pathway_synapses_4(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses(None, [1, 2], None),
            [0, 1, 2, 3]
        )

    def test_pathway_synapses_5(self):
        pytest.raises(
            BluePyError,
            self.test_obj.pathway_synapses, None, None, None
        )

    def test_pathway_synapses_6(self):
        npt.assert_equal(
            self.test_obj.pathway_synapses([], [], None),
            []
        )

    def test_pathway_synapses_7(self):
        assert self.test_obj.pathway_synapses([], [], Synapse.AXONAL_DELAY).empty

    def test_iter_connections_1(self):
        it = self.test_obj.iter_connections(
            [1, 3], [2]
        )
        assert next(it) == (1, 2)
        assert next(it) == (3, 2)
        pytest.raises(StopIteration, next, it)

    def test_iter_connections_2(self):
        it = self.test_obj.iter_connections(
            [1, 3], [2], unique_gids=True
        )
        assert list(it) == [(1, 2)]

    def test_iter_connections_3(self):
        it = self.test_obj.iter_connections(
            [1, 3], [2], shuffle=True
        )
        assert sorted(it) == [(1, 2), (3, 2)]

    def test_iter_connections_4(self):
        it = self.test_obj.iter_connections(
            None, None
        )
        pytest.raises(BluePyError, next, it)

    def test_iter_connections_5(self):
        it = self.test_obj.iter_connections(
            None, [2]
        )
        assert list(it) == [(1, 2), (3, 2)]

    def test_iter_connections_6(self):
        it = self.test_obj.iter_connections(
            [3], None
        )
        assert list(it) == [(3, 1), (3, 2)]

    def test_iter_connections_7(self):
        it = self.test_obj.iter_connections(
            [1, 3], [2], return_synapse_ids=True
        )
        npt.assert_equal(list(it), [(1, 2, [1, 2]), (3, 2, [3])])

    def test_iter_connections_8(self):
        it = self.test_obj.iter_connections(
            [1, 3], [2], return_synapse_count=True
        )
        assert list(it) == [(1, 2, 2), (3, 2, 1)]

    def test_iter_connections_9(self):
        pytest.raises(
            BluePyError,
            self.test_obj.iter_connections,
            [1, 3], [2], return_synapse_ids=True, return_synapse_count=True
        )

    def test_iter_connections_10(self):
        it = self.test_obj.iter_connections(
            [], [1, 2, 3]
        )
        assert list(it) == []

    @patch.dict(sys.modules, {'libFLATIndex': 'foo'})
    def test_spatial_index(self):
        with pytest.raises(BluePyDeprecationError):
            self.test_obj.spatial_index

    def test_iter_connections_11(self):
        it = self.test_obj.iter_connections(
            [1, 2, 3], []
        )
        assert list(it) == []

    def test_iter_connections_12(self):
        circuit = Mock()
        circuit.cells.ids = lambda x: x
        test_obj = Connectome(os.path.join(TEST_DATA_DIR, "edges_complete_graph.sonata"), circuit)
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

        it = test_obj.iter_connections([2, 3], [1, 2, 3], unique_gids=True)
        assert sorted(it) == [(2, 1), (3, 2)]

        it = test_obj.iter_connections([1, 2, 3], [2, 3], unique_gids=True)
        assert sorted(it) == [(1, 2), (2, 3)]

    def test_old_new_format_compatibility(self):
        properties = [Synapse.PRE_GID,
                      Synapse.POST_GID,
                      Synapse.AXONAL_DELAY,
                      Synapse.D_SYN,
                      Synapse.DTC,
                      Synapse.F_SYN,
                      Synapse.G_SYNX,
                      Synapse.TYPE,
                      Synapse.U_SYN,
                      Synapse.PRE_SECTION_ID,
                      Synapse.POST_BRANCH_TYPE,
                      Synapse.POST_SECTION_ID,
                      Synapse.PRE_X_CENTER,
                      ]
        result_new = self.test_obj.synapse_properties([2], properties)

        test_obj_old = Connectome(os.path.join(TEST_DATA_DIR, "edges_old.sonata"),
                                  self.circuit)
        result_old = test_obj_old.synapse_properties([2], properties)
        for prop in properties:
            new = result_new[prop].iloc[0]
            old = result_old[prop].iloc[0]
            if isinstance(new, np.floating):
                check = partial(npt.assert_almost_equal, decimal=5)
            else:
                check = npt.assert_equal
            check(new, old, err_msg="{0}: {1} != {2}".format(prop, new, old))


class TestSonataConnectomeSonataCells(TestSonataConnectomeMvdCells):
    def setup_method(self):
        self.circuit = Mock()
        self.circuit.morph.section_features.return_value = pd.DataFrame({
            Section.NEURITE_START_DISTANCE: [100., 200.],
        }, index=pd.Index([1, 2], name=Section.ID))
        self.circuit.morph.segment_features.return_value = pd.DataFrame({
            Segment.SECTION_START_DISTANCE: [10., 20., 30.],
        }, index=pd.MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)],
                                           names=[Section.ID, Segment.ID]))
        self.circuit.cells = CellCollection(os.path.join(TEST_DATA_DIR, "circuit.h5"))
        metadata = {"prop1": "string_prop", "prop2": 1}
        self.test_obj = Connectome(os.path.join(TEST_DATA_DIR, "edges.sonata"),
                                   self.circuit, metadata=metadata)
