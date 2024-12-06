import os
import re
from unittest.mock import Mock, patch

import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from utils import copy_file, tmp_file

import bluepy.impl.cells_sonata as test_module
from bluepy import Circuit
from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.impl.target import TargetContext

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test__euler2mat():
    pi2 = 0.5 * np.pi
    actual = test_module._euler2mat(
        [0.0, pi2],  # rotation_angle_z
        [pi2, 0.0],  # rotation_angle_y
        [pi2, pi2],  # rotation_angle_x
    )
    expected = np.array(
        [
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    npt.assert_almost_equal(actual, expected)

    with pytest.raises(BluePyError):
        test_module._euler2mat([pi2, pi2], [pi2, pi2], [pi2])  # ax|y|z not of same size


def test__quaternion2mat():
    actual = test_module._quaternion2mat(
        [1, 1, 1],
        [
            1,
            0,
            0,
        ],
        [0, 1, 0],
        [0, 0, 1],
    )
    expected = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
    )
    npt.assert_almost_equal(actual, expected)

class TestSonataCellCollection:
    def setup_method(self):
        # this sonata file is created from the circuit.mvd3 using brainbuilder
        sonata_path = os.path.join(TEST_DATA_DIR, "circuit.sonata")
        targets = TargetContext.load([os.path.join(TEST_DATA_DIR, "E.target")])
        self.cells = test_module.SonataCellCollection(sonata_path, targets)

    def test_multiple_pop_fails(self):
        sonata_path = os.path.join(TEST_DATA_DIR, "circuit.sonata")
        targets = {
            'A': np.array([1, 2]),
            'B': np.array([9999]),
            'C': np.empty(0, dtype=int)
        }
        targets_mock = Mock()
        targets_mock.resolve = lambda name: targets[name]

        with patch('libsonata.NodeStorage.population_names', return_values=["1", "2"]):
            with pytest.raises(BluePyError):
                test_module.SonataCellCollection(sonata_path, targets_mock)

    def test_available_properties(self):
        res = self.cells.available_properties
        expected = {'etype', 'synapse_class', 'morphology', '@dynamics:propB',
                    'rotation_angle_xaxis', 'morph_class', 'x', 'layer', 'rotation_angle_zaxis',
                    'z', 'model_template', '@dynamics:propA', 'mtype', 'rotation_angle_yaxis',
                    'minicolumn', 'hypercolumn', 'y', 'orientation'}
        missing = res - expected
        npt.assert_equal(len(missing), 0)

    def test_available_properties_quaternions(self):
        default_file = os.path.join(TEST_DATA_DIR, "circuit_quaternions.sonata")
        with copy_file(default_file) as filepath:
            cells = test_module.SonataCellCollection(filepath, None)
            res = cells.available_properties
            expected = {'etype', 'synapse_class', 'morphology', '@dynamics:propB',
                        'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
                        'morph_class', 'x', 'layer', 'z', 'model_template', '@dynamics:propA', 'mtype',
                        'minicolumn', 'hypercolumn', 'y', 'orientation'}
            missing = res - expected
            npt.assert_equal(len(missing), 0)

    def test_orientation_partial_euler(self):
        default_file = os.path.join(TEST_DATA_DIR, "circuit.sonata")
        with copy_file(default_file) as filepath:
            with h5py.File(filepath, "r+") as h5:
                del h5[f"nodes/default/0/rotation_angle_yaxis"]
            cells = test_module.SonataCellCollection(filepath, None)
            res = cells.available_properties
            expected = {'etype', 'synapse_class', 'morphology', '@dynamics:propB',
                        'rotation_angle_xaxis', 'morph_class', 'x', 'layer', 'rotation_angle_zaxis',
                        'z', 'model_template', '@dynamics:propA', 'mtype',
                        'minicolumn', 'hypercolumn', 'y', 'orientation'}
            missing = res - expected
            npt.assert_equal(len(missing), 0)

    def test_orientation_fail(self):
        default_file = os.path.join(TEST_DATA_DIR, "circuit_quaternions.sonata")

        # If quaternions you need all 4 coordinates
        with copy_file(default_file) as filepath:
            with h5py.File(filepath, "r+") as h5:
                del h5[f"nodes/default/0/orientation_w"]
            with pytest.raises(BluePyError):
                test_module.SonataCellCollection(filepath, None)

        # Cannot have both quaternions and a eulers at the same time
        with copy_file(default_file) as filepath:
            with h5py.File(filepath, "r+") as h5:
                h5.create_dataset(f"nodes/default/0/rotation_angle_zaxis", data=[1,2,3,4])
            with pytest.raises(BluePyError):
                test_module.SonataCellCollection(filepath, None)

    def test_ids(self):
        npt.assert_equal(self.cells.ids(), [1, 2, 3])
        npt.assert_equal(self.cells.ids(group={}), [1, 2, 3])
        npt.assert_equal(self.cells.ids(group=[]), [])
        npt.assert_equal(self.cells.ids(limit=1), [1])
        npt.assert_equal(len(self.cells.ids(sample=2)), 2)
        npt.assert_equal(self.cells.ids(1), [1])
        npt.assert_equal(self.cells.ids([1, 2]), [1, 2])
        npt.assert_equal(self.cells.ids([2, 1, 2]), [2, 1, 2])  # order and duplicates preserved
        npt.assert_equal(self.cells.ids(np.array([2, 1, 2])), np.array([2, 1, 2]))
        npt.assert_equal(self.cells.ids({Cell.MTYPE: 'L6_Y'}), [2, 3])
        npt.assert_equal(self.cells.ids('A'), [1, 2])
        npt.assert_equal(self.cells.ids({'$target': 'A'}), [1, 2])
        npt.assert_equal(self.cells.ids({'$target': 'A', Cell.MTYPE: 'L6_Y'}), [2])
        npt.assert_equal(self.cells.ids({'$target': 'C'}), [])
        npt.assert_equal(self.cells.ids(999), [999])  # no 999 GID but check is removed
        npt.assert_equal(self.cells.ids([1, 999]), [1, 999])  # no 999 GID but check is removed
        npt.assert_equal(self.cells.ids(group=[], sample=15), [])
        npt.assert_equal(self.cells.ids(group={Cell.MTYPE: "unknown"}, sample=15), [])

    def test_ids_raises_exception(self):
        with pytest.raises(
            BluePyError, match="Could not apply properties filter to external target"
        ):
            self.cells.ids({'$target': 'B', Cell.MTYPE: 'L6_Y'})
        with pytest.raises(BluePyError, match="Target NONEXISTENT doesn't exist"):
            self.cells.ids({'$target': 'NONEXISTENT', Cell.MTYPE: 'L6_Y'})
        with pytest.raises(BluePyError, match=re.escape("Unknown cell properties: [$err]")):
            self.cells.ids({'$err': 0})

    def test_get(self):
        # more values when convert the mvd3 to sonata --> add 2 dynamics + 3 rotation angles +
        # orientation 12 + 5 + 1 = 18
        assert self.cells.get().shape == (3, 18)
        assert self.cells.get(1, Cell.MTYPE) == 'L2_X'
        assert self.cells.get(np.int32(1), Cell.MTYPE) == 'L2_X'

        pdt.assert_series_equal(
            self.cells.get([1], properties=Cell.MTYPE),
            pd.Series(['L2_X'], index=[1], name='mtype')
        )
        pdt.assert_frame_equal(
            self.cells.get([2, 3], properties=[Cell.X, Cell.Y, Cell.Z]),
            pd.DataFrame([
                [201., 202., 203.],
                [301., 302., 303.],
            ],
                columns=[Cell.X, Cell.Y, Cell.Z],
                index=[2, 3]
            )
        )
        pytest.raises(BluePyError, self.cells.get, 999)  # no such GID: 999
        pytest.raises(BluePyError, self.cells.get, [1, 999])  # no such GID: 999
        expected = np.array([[[0.73821992, 0., 0.67456012],
                              [0., 1., 0.],
                              [-0.67456012, 0., 0.73821992]],
                             [[-0.86768965, -0.44169042, 0.22808825],
                              [0.48942842, -0.8393853, 0.23641518],
                              [0.0870316, 0.31676788, 0.94450178]],
                             [[0.46298666, 0., 0.88636525],
                              [0., 1., 0.],
                              [-0.88636525, 0., 0.46298666]]])

        res = self.cells.get(properties=Cell.ORIENTATION)
        npt.assert_allclose(res.loc[1], expected[0])
        npt.assert_allclose(res.loc[2], expected[1])
        npt.assert_allclose(res.loc[3], expected[2])
        npt.assert_allclose(self.cells.get(group=2, properties=Cell.ORIENTATION),
                            expected[1])

        # check if the orientation is propagated to the get by default
        res = self.cells.get()[Cell.ORIENTATION]
        npt.assert_allclose(res.loc[1], expected[0])
        npt.assert_allclose(res.loc[2], expected[1])
        npt.assert_allclose(res.loc[3], expected[2])

        with pytest.raises(BluePyError):
            self.cells.get(properties="unknown")


def test_trigger_rotation_with_quaternions():
    """Test that rotations are computed with quaternions in SonataCellCollection._data."""
    # synapse_class has 2 enum values ['EXC', 'INH'] but three nodes
    # morph_class has 1 enum value ['INT'] but three nodes
    sonata_path = os.path.join(TEST_DATA_DIR, "circuit_quaternions.sonata")
    cells = test_module.SonataCellCollection(sonata_path)
    assert cells._data.shape == (3, 19)


def test_using_categoricals():
    """Test different uses of categoricals in SonataNodePopulation._data"""
    sonata_path = os.path.join(TEST_DATA_DIR, "circuit_categoricals.sonata")
    cells = test_module.SonataCellCollection(sonata_path)
    assert cells._data.shape == (3, 19)


def test_ids_projection():
    """Test that projection ids are returned [BLPY-265]."""
    content = """
Run Default
    {{
        CircuitPath {dir}
        METypePath {dir}
        CellLibraryFile {dir}/circuit.sonata
        nrnPath {dir}/edges.sonata
        TargetFile {dir}/projection.target
        MorphologyPath {dir}/morphs
        MorphologyType h5
    }}

Projection Test
    {{
    Path {dir}/projection.sonata
    Source Test
    PopulationID 1
    Type Synaptic
    }}
        """.format(dir=TEST_DATA_DIR)
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        circuit = Circuit(filepath)
        npt.assert_array_equal(circuit.cells.ids('Test'), [10, 11, 12])
        with pytest.raises(BluePyError, match='Could not apply properties filter to external target'):
            circuit.cells.ids({'$target': 'Test', 'some_attr': 'some_value'})
