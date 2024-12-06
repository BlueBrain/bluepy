import os
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepy.impl.cells_mvd as test_module
from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.utils import deprecate

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_load_invalid_extension():
    pytest.raises(
        BluePyError,
        test_module.load, 'some_file.xls'
    )


def test_load_MVD2():
    cells = test_module.load(os.path.join(TEST_DATA_DIR, "circuit.mvd2"))
    assert cells.index.tolist() == [1, 2, 3]
    expected = pd.Series({
        Cell.ETYPE: 'bNA',
        Cell.HYPERCOLUMN: 0,
        Cell.LAYER: 6,
        Cell.MINICOLUMN: 0,
        Cell.MORPH_CLASS: 'INT',
        Cell.MORPHOLOGY: 'morph-B',
        Cell.ME_COMBO: 'me-combo-B',
        Cell.MTYPE: 'L6_Y',
        Cell.ORIENTATION: [
            [0.609868565477, 0.0, 0.792502575922],
            [0.0, 1.0, 0.0],
            [-0.792502575922, 0.0, 0.609868565477]
        ],
        Cell.SYNAPSE_CLASS: 'INH',
        Cell.X: 201.,
        Cell.Y: 202.,
        Cell.Z: 203.,
    }, name=2)
    actual = cells.loc[2].sort_index()
    expected = expected.sort_index()
    pdt.assert_series_equal(actual, expected)


def test_load_MVD3():
    cells = test_module.load(os.path.join(TEST_DATA_DIR, "circuit.mvd3"))
    assert cells.index.tolist() == [1, 2, 3]
    expected = pd.Series({
        Cell.ETYPE: 'bNA',
        Cell.HYPERCOLUMN: 0,
        Cell.LAYER: 6,
        Cell.MINICOLUMN: 0,
        Cell.MORPH_CLASS: 'INT',
        Cell.MORPHOLOGY: 'morph-B',
        Cell.ME_COMBO: 'me-combo-B',
        Cell.MTYPE: 'L6_Y',
        Cell.ORIENTATION: [
            [-0.86768965, -0.44169042, 0.22808825],
            [0.48942842, -0.8393853, 0.23641518],
            [0.0870316, 0.31676788, 0.94450178],
        ],
        Cell.SYNAPSE_CLASS: 'INH',
        Cell.X: 201.,
        Cell.Y: 202.,
        Cell.Z: 203.,
    }, name=2)
    actual = cells.loc[2].sort_index()
    expected = expected.sort_index()
    pdt.assert_series_equal(actual, expected)


def test_gids_by_filter():
    cells = pd.DataFrame({
        Cell.X: [0.0, 0.5, 1.0],
        Cell.MTYPE: pd.Categorical.from_codes([0, 1, 1], ['A', 'B', 'C']),
        Cell.LAYER: [1, 2, 3],
    })
    pytest.raises(
        BluePyError,
        test_module.gids_by_filter, cells, {'err': 23}
    )
    npt.assert_equal(
        [],
        test_module.gids_by_filter(cells, {Cell.MTYPE: 'err'})
    )
    npt.assert_equal(
        [1],
        test_module.gids_by_filter(cells, {
            Cell.X: (0, 0.7),
            Cell.MTYPE: ['B', 'C'],
            Cell.LAYER: (1, 2)
        })
    )


def test_gids_by_filter_complex_query():
    cells = pd.DataFrame({
        Cell.MTYPE: ['L23_MC', 'L4_BP', 'L6_BP', 'L6_BPC'],
    })
    # only full match is accepted
    npt.assert_equal(
        [1, 2],
        test_module.gids_by_filter(cells, {
            Cell.MTYPE: {'$regex': '.*BP'},
        })
    )
    # ...not 'startswith'
    npt.assert_equal(
        [],
        test_module.gids_by_filter(cells, {
            Cell.MTYPE: {'$regex': 'L6'},
        })
    )
    # ...or 'endswith'
    npt.assert_equal(
        [],
        test_module.gids_by_filter(cells, {
            Cell.MTYPE: {'$regex': 'BP'},
        })
    )

    with pytest.raises(deprecate.BluePyDeprecationError):
        test_module.gids_by_filter(cells, {Cell.MTYPE: '@.*BP',})

    with pytest.raises(deprecate.BluePyDeprecationError):
        test_module.gids_by_filter(cells, {Cell.MTYPE: 'regex:.*BP'})

    # '$regex' is the only query modifier supported for the moment
    with pytest.raises(BluePyError):
        test_module.gids_by_filter(cells, {Cell.MTYPE: {'err': '.*BP'}})


class TestMVDCellCollection:

    def setup_method(self):
        mvd_path = os.path.join(TEST_DATA_DIR, "circuit.mvd3")
        targets = {
            'A': np.array([1, 2]),
            'B': np.array([9999]),
            'C': np.empty(0, dtype=int)
        }
        targets_mock = Mock()
        targets_mock.resolve = lambda name: targets[name]
        self.cells = test_module.MVDCellCollection(mvd_path, targets_mock)

    def test_available_properties(self):
        res = self.cells.available_properties
        expected = {'minicolumn', 'synapse_class', 'layer', 'x', 'morphology', 'mtype',
                    'orientation', 'y', 'morph_class', 'z', 'etype', 'me_combo', 'hypercolumn'}
        missing = res - expected
        npt.assert_equal(len(missing), 0)

    def test_ids(self):
        npt.assert_equal(self.cells.ids(), [1, 2, 3])
        npt.assert_equal(self.cells.ids({}), [1, 2, 3])
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
        npt.assert_equal(self.cells.ids({'$target': 'B'}), [9999])
        pytest.raises(BluePyError, self.cells.ids, {'$target': 'B', Cell.MTYPE: 'L6_Y'})
        npt.assert_equal(self.cells.ids({'$target': 'C'}), [])
        pytest.raises(BluePyError, self.cells.ids, {'$err': 0})
        npt.assert_equal(self.cells.ids(999), [999])  # no 999 GID but check is removed
        npt.assert_equal(self.cells.ids([1, 999]), [1, 999])  # no 999 GID but check is removed
        npt.assert_equal(self.cells.ids(group=[], sample=15), [])
        npt.assert_equal(self.cells.ids(group={Cell.MTYPE: "unknown"}, sample=15), [])

    def test_get(self):
        assert self.cells.get().shape == (3, 13)
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
