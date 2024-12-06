import os
import pickle
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from bluepy_configfile.configfile import BlueConfig
from utils import setup_tempdir, tmp_file

import bluepy.circuit as test_module
from bluepy.circuit_stats import StatsHelper
from bluepy.connectome import Connectome
from bluepy.emodels import EModelHelper
from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.morphology import MorphHelper
from bluepy.subcellular import SubcellularHelper

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

TEST_BLUECONFIG = os.path.join(TEST_DATA_DIR, "CircuitConfig")


def test_find_cells_1():
    blueconfig = MagicMock()
    m = Mock(spec=['CircuitPath'])
    blueconfig.Run = m
    blueconfig.Run.CircuitPath = '/aaa'
    with patch('os.path.exists', side_effect=[True]):
        assert test_module._find_cells(blueconfig) == '/aaa/circuit.mvd3'


def test_find_cells_2():
    blueconfig = MagicMock()
    m = Mock(spec=['CircuitPath'])
    blueconfig.Run = m
    blueconfig.Run.CircuitPath = '/aaa'
    with patch('os.path.exists', side_effect=[False, True]):
        assert test_module._find_cells(blueconfig) == '/aaa/circuit.mvd2'


def test_find_cells_3():
    blueconfig = Mock()
    blueconfig.Run.CircuitPath = '/aaa'
    blueconfig.Run.CellLibraryFile = '/abspath/my_file.h5'
    with patch('os.path.exists', side_effect=[True]):
        assert test_module._find_cells(blueconfig) == '/abspath/my_file.h5'


def test_find_cells_4():
    blueconfig = Mock()
    blueconfig.Run.CircuitPath = '/aaa'
    blueconfig.Run.CellLibraryFile = '/abspath/my_file.h5'
    with patch('os.path.exists', side_effect=[False, True]):
        assert test_module._find_cells(blueconfig) == '/aaa/circuit.mvd3'


def test_find_cells_5():
    blueconfig = MagicMock()
    m = Mock(spec=['CircuitPath'])
    blueconfig.Run = m
    blueconfig.Run.CircuitPath = '/aaa'
    with patch('os.path.exists', side_effect=[False, False]):
        pytest.raises(
            BluePyError,
            test_module._find_cells, blueconfig
        )


def test_find_cells_6():
    blueconfig = Mock()
    blueconfig.Run.CircuitPath = '/aaa'
    blueconfig.Run.CellLibraryFile = 'circuit.mvd3'
    with patch('os.path.exists', side_effect=[True]):
        assert test_module._find_cells(blueconfig) == '/aaa/circuit.mvd3'


def test_find_targets_1():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    blueconfig.Run.CircuitPath = '/bbb'
    blueconfig.Run.__contains__ = lambda *args: False
    with patch('os.path.exists', side_effect=[True]):
        assert test_module._find_targets(blueconfig) == ['/aaa/start.target']


def test_find_targets_2():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    blueconfig.Run.CircuitPath = '/bbb'
    blueconfig.Run.__contains__ = lambda *args: False
    with patch('os.path.exists', side_effect=[False, True]):
        assert test_module._find_targets(blueconfig) == ['/bbb/start.target']


def test_find_targets_3():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    blueconfig.Run.CircuitPath = '/bbb'
    blueconfig.Run.__contains__ = lambda *args: False
    with patch('os.path.exists', side_effect=[False, False]):
        pytest.raises(
            BluePyError,
            test_module._find_targets, blueconfig
        )


def test_find_targets_4():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    blueconfig.Run.CircuitPath = '/bbb'
    blueconfig.Run.TargetFile = '/ccc/qqq.target'
    blueconfig.Run.__contains__ = lambda *args: True
    with patch('os.path.exists', side_effect=[True, True]):
        assert test_module._find_targets(blueconfig) == ['/aaa/start.target', '/ccc/qqq.target']


def test_find_targets_5():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    blueconfig.Run.CircuitPath = '/bbb'
    blueconfig.Run.TargetFile = '/ccc/qqq.target'
    blueconfig.Run.__contains__ = lambda *args: True
    with patch('os.path.exists', side_effect=[True, False]):
        pytest.raises(
            BluePyError,
            test_module._find_targets, blueconfig
        )


def test_find_spatial_index_1():
    exists = {
        '/aaa/test_index.dat': True,
        '/aaa/test_index.idx': True,
        '/aaa/test_payload.dat': True,
    }
    with patch('os.path.exists', side_effect=exists.__getitem__):
        assert test_module._find_spatial_index('/aaa', 'test') == '/aaa'


def test_find_spatial_index_2():
    exists = {
        '/aaa/test_index.dat': True,
        '/aaa/test_index.idx': True,
        '/aaa/test_payload.dat': False,
    }
    with patch('os.path.exists', side_effect=exists.__getitem__):
        assert test_module._find_spatial_index('/aaa', 'test') is None


def test_find_segment_index():
    blueconfig = Mock()
    blueconfig.Run.CircuitPath = '/aaa'
    exists = {
        '/aaa/SEGMENT_index.dat': True,
        '/aaa/SEGMENT_index.idx': True,
        '/aaa/SEGMENT_payload.dat': True,
    }
    with patch('os.path.exists', side_effect=exists.__getitem__):
        assert test_module._find_segment_index(blueconfig) == '/aaa'


def test_find_synapse_index_1():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa'
    exists = {
        '/aaa/SYNAPSE_index.dat': True,
        '/aaa/SYNAPSE_index.idx': True,
        '/aaa/SYNAPSE_payload.dat': True,
    }
    with patch('os.path.exists', side_effect=exists.__getitem__):
        with patch('os.path.isfile', return_value=False):
            assert test_module._find_synapse_index(blueconfig) == '/aaa'


def test_find_synapse_index_2():
    blueconfig = Mock()
    blueconfig.Run.nrnPath = '/aaa/edges.sonata'
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isfile', return_value=True):
            assert test_module._find_synapse_index(blueconfig) == '/aaa'


def test_resolve_paths():
    actual = {
        'foo1': 'bar1',
        'foo2': './bar2',
        'foo3': 42,
        'foo4': '.',
        'foo5': {
            'foo51': './bar51'
        },
        'foo6': [
            './bar61', 'bar62'
        ]
    }
    test_module._resolve_paths(actual, circuit_dir="/test/")
    expected = {
        'foo1': 'bar1',
        'foo2': '/test/bar2',
        'foo3': 42,
        'foo4': '.',
        'foo5': {
            'foo51': '/test/bar51'
        },
        'foo6': [
            '/test/bar61', 'bar62'
        ]
    }
    assert actual == expected


def test_open_circuit():
    # Can open BlueConfig file or use BlueConfig object
    with patch('os.path.exists'):
        test_module.Circuit(TEST_BLUECONFIG)
        with open(TEST_BLUECONFIG) as f:
            test_module.Circuit(BlueConfig(f))

    # Invalid config type
    pytest.raises(BluePyError, test_module.Circuit, 42)


class TestCircuit:
    def setup_method(self):
        self.circuit = test_module.Circuit({
            'cells': os.path.join(TEST_DATA_DIR, "circuit.mvd2"),
            'morphologies': TEST_DATA_DIR,
            'emodels': "stub",
            'connectome': os.path.join(TEST_DATA_DIR, "nrn"),
            'projections': {
                'projA': os.path.join(TEST_DATA_DIR, "nrn"),
            },
            'projections_metadata': {
                'projA': {"Source": "source", "PopulationID": 1},
            },
            'targets': [
                os.path.join(TEST_DATA_DIR, "B.target")
            ],
            'soma_index': "stub",
            'synapse_index': "stub",
            'subcellular': "stub",
            'atlas': "stub",
        })

    @patch.dict(sys.modules, {'libFLATIndex': 'foo'})
    def test_cells(self):
        cells = self.circuit.cells
        assert cells.ids({Cell.LAYER: 6}, limit=1) == [2]
        assert cells.get(1, Cell.LAYER) == 2
        pdt.assert_frame_equal(
            cells.positions([1, 3]),
            pd.DataFrame([
                [101.0, 102.0, 103.0],
                [301.0, 302.0, 303.0],
            ], index=[1, 3], columns=['x', 'y', 'z'])
        )
        pdt.assert_series_equal(
            cells.positions(2),
            pd.Series([
                201.0, 202.0, 203.0,
            ], name=2, index=['x', 'y', 'z'])
        )
        pdt.assert_series_equal(
            cells.orientations([1, 3]),
            pd.Series([
                [[0.7382199, 0., 0.6745601], [0., 1., 0.], [-0.6745601, 0., 0.7382199]],
                [[0.4629866, 0., 0.8863652], [0., 1., 0.], [-0.8863652, 0., 0.4629866]],
            ], name=Cell.ORIENTATION, index=[1, 3])
        )
        npt.assert_almost_equal(
            cells.orientations(2),
            [[0.6098685, 0.0, 0.7925025], [0.0, 1.0, 0.0], [-0.7925025, 0.0, 0.6098685]]
        )
        assert cells.count() == 3
        assert cells.mtypes == {'L2_X', 'L6_Y'}
        assert cells.etypes == {'bNA', 'cNA'}
        assert cells.targets == {'X', 'Z'}
        assert cells.spatial_index.exists() == False

    def test_config(self):
        assert self.circuit.config == {
            'cells': os.path.join(TEST_DATA_DIR, "circuit.mvd2"),
            'morphologies': TEST_DATA_DIR,
            'emodels': "stub",
            'connectome': os.path.join(TEST_DATA_DIR, "nrn"),
            'projections': {
                'projA': os.path.join(TEST_DATA_DIR, "nrn"),
            },
            'projections_metadata': {
                'projA': {"Source": "source", "PopulationID": 1},
            },
            'targets': [
                os.path.join(TEST_DATA_DIR, "B.target")
            ],
            'soma_index': "stub",
            'synapse_index': "stub",
            'subcellular': "stub",
            'atlas': "stub",
        }

    def test_config_blueconfig(self):
        def _test_config(morph_type):
            with setup_tempdir() as tmp:
                simple_blueconfig = """
Run Default
{{
    CircuitPath {}
    nrnPath {}
    CellLibraryFile circuit.mvd3
    METypePath {}
    MorphologyPath {}
    {} {}
}}
                        """.format(TEST_DATA_DIR,
                                   TEST_DATA_DIR,
                                   TEST_DATA_DIR,
                                   TEST_DATA_DIR,
                                   "MorphologyType" if morph_type else '',
                                   morph_type if morph_type else '')
                config_path = os.path.join(tmp, 'CircuitConfig')
                with open(config_path, "w") as conf:
                    conf.write(simple_blueconfig)
                circuit = test_module.Circuit(config_path)
                expected = {'cells': os.path.join(TEST_DATA_DIR, "circuit.mvd3"),
                            'morphologies': TEST_DATA_DIR, 'emodels': TEST_DATA_DIR,
                            'connectome': TEST_DATA_DIR,
                            'targets': [os.path.join(TEST_DATA_DIR, 'start.target')],
                            'projections': {},
                            'projections_metadata': {}, 'segment_index': None,
                            'synapse_index': None,
                            'subcellular': os.path.join(TEST_DATA_DIR, 'subcellular.h5'),
                            'morphology_type': morph_type
                            }
                if not morph_type:
                    expected.pop('morphology_type')
                assert circuit.config == expected

        _test_config(None)
        _test_config("asc")

    def test_morph(self):
        assert isinstance(self.circuit.morph, MorphHelper)

    def test_emodels(self):
        assert isinstance(self.circuit.emodels, EModelHelper)

    @patch.dict(sys.modules, {'libFLATIndex': 'foo'})
    def test_connectome(self):
        assert isinstance(self.circuit.connectome, Connectome)
        self.circuit.connectome.spatial_index.exists()

    def test_projection(self):
        assert isinstance(self.circuit.projection('projA'), Connectome)
        assert isinstance(self.circuit.projection('projA'), Connectome)  # check caching

    def test_projection_2(self):
        proj = self.circuit.projection('projA')
        assert proj.metadata == {"Source": "source", "PopulationID": 1}

    def test_projection_3(self):
        # non regression test for old circuit instantiate with a Mapping object
        circuit = test_module.Circuit({
            'cells': os.path.join(TEST_DATA_DIR, "circuit.mvd2"),
            'morphologies': TEST_DATA_DIR,
            'emodels': "stub",
            'connectome': os.path.join(TEST_DATA_DIR, "nrn"),
            'projections': {
                'projA': os.path.join(TEST_DATA_DIR, "nrn"),
            },
            'targets': [
                os.path.join(TEST_DATA_DIR, "B.target")
            ],
            'soma_index': "stub",
            'synapse_index': "stub",
            'subcellular': "stub",
            'atlas': "stub",
        })
        proj = circuit.projection('projA')
        assert proj.metadata is None

    def test_stats(self):
        assert isinstance(self.circuit.stats, StatsHelper)

    def test_subcellular(self):
        assert isinstance(self.circuit.subcellular, SubcellularHelper)

    def test_atlas(self):
        from voxcell.nexus.voxelbrain import Atlas
        assert isinstance(self.circuit.atlas, Atlas)

    def test_pickle_roundtrip(self):
        dumped = pickle.dumps(self.circuit)
        loaded = pickle.loads(dumped)

        assert isinstance(loaded.config, dict)
        assert loaded.config == self.circuit.config


def test_circuit_colon():
    content = """
Run Default
    {{
        CircuitPath {dir}
        METypePath {dir}
        CellLibraryFile {dir}/circuit.sonata
        nrnPath {dir}/multi_pop_edge.h5:default2

        MorphologyPath {dir}/morphs
        MorphologyType h5
        TargetFile {dir}/colon.target
        OutputRoot {dir}
    }}
    """.format(dir=TEST_DATA_DIR)
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        circuits = test_module.Circuit(filepath)
        assert circuits.connectome._impl._population.name == "default2"
        assert circuits.cells.targets == {'default:A', 'A', 'default2:A'}
        npt.assert_equal(circuits.cells.ids("A"), [1, 2])
        npt.assert_equal(circuits.cells.ids("default:A"), [1, 3])
        npt.assert_equal(circuits.cells.ids("default2:A"), [1])


def test_circuit_multi_pop_no_colon_fails():
    content = """
Run Default
    {{
        CircuitPath {dir}
        METypePath {dir}
        CellLibraryFile {dir}/circuit.sonata
        nrnPath {dir}/multi_pop_edge.h5

        MorphologyPath {dir}/morphs
        MorphologyType h5
        TargetFile {dir}/colon.target
        OutputRoot {dir}
    }}
    """.format(dir=TEST_DATA_DIR)
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        circuits = test_module.Circuit(filepath)
        with pytest.raises(BluePyError) as e:
            circuits.connectome
            assert "Only single-population" in str(e.value)


def test_circuit_colon_nrn():
    content = """
Run Default
    {{
        CircuitPath {dir}
        METypePath {dir}
        CellLibraryFile {dir}/circuit.sonata
        nrnPath {dir}/nrn/nrn.h5:default2

        MorphologyPath {dir}/morphs
        MorphologyType h5
        TargetFile {dir}/E.target
        OutputRoot {dir}
    }}
    """.format(dir=TEST_DATA_DIR)
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        circuits = test_module.Circuit(filepath)
        npt.assert_equal(circuits.cells.ids(), [1, 2, 3])
        with pytest.raises(BluePyError) as e:
            circuits.connectome   # proc the connectome creation
            assert "Could not define a population" in str(e.value)
