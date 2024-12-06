import os
from unittest.mock import Mock, patch

import matplotlib
import numpy.testing as npt
import pytest
from utils import PatchImport

import bluepy.plotting as test_module
from bluepy.circuit import Circuit
from bluepy.exceptions import BluePyError
from bluepy.impl.compartment_report import SomaReport

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def choice(result, sample, replace=False):
    return result[1]


def report(filepath):
    return SomaReport(os.path.join(TEST_DATA_DIR, filepath))


class TestPlotting:
    def setup_method(self):
        circuit = Circuit({
            'cells': os.path.join(TEST_DATA_DIR, "circuit.mvd3"),
            'targets': [
                os.path.join(TEST_DATA_DIR, "A.target")
            ]
        })
        self.sim = Mock()
        self.sim.circuit = circuit
        self.sim.t_start = 0.0
        self.sim.t_end = 1000
        self.sim.target_gids = circuit.cells.ids()
        self.sim.report = report

    @patch('numpy.random.choice', choice)
    def test_select_gids(self):
        npt.assert_array_equal(test_module.select_gids(self.sim, None, None), [1, 2, 3])
        npt.assert_array_equal(test_module.select_gids(self.sim, 'Y', None), [3])
        npt.assert_array_equal(test_module.select_gids(self.sim, 'X', None), [1, 2])
        npt.assert_array_equal(test_module.select_gids(self.sim, 'X', 1), [2])
        with patch('bluepy.plotting.L') as mock_logger:
            npt.assert_array_equal(test_module.select_gids(self.sim, [1, 2, 3, 4], None), [1, 2, 3])
            assert mock_logger.warning.call_count == 1
        npt.assert_array_equal(test_module.select_gids(self.sim, [10, 20], None,
                                                       report_name='soma.h5'), [10, 20])

    def test_group_to_str_1(self):
        assert test_module.group_to_str('target') == 'target'
        assert test_module.group_to_str('target', sample=15) == 'target, sample = 15'
        assert test_module.group_to_str(15) == '15'
        assert test_module.group_to_str([1 == 2, 3, 4, 5, 6],
                        "[1, 2, 3, 4, 5, '...']")
        assert test_module.group_to_str([1, 2, 3, 4, 5, 6], nb_max=2) == "[1, 2, '...']"
        assert test_module.group_to_str("") == ""
        assert test_module.group_to_str(None) == ""
        assert test_module.group_to_str({"x": (1, 2)}) == "{'x': (1, 2)}"
        assert test_module.group_to_str({"mtype": "L2"}) == "{'mtype': 'L2'}"

    def test_group_to_str_2(self):
        with pytest.raises(BluePyError):
            test_module.group_to_str(tuple())

    def test_check_times_1(self):
        assert test_module.check_times(self.sim, t_start=15, t_end=500) == (15, 500)
        assert test_module.check_times(self.sim, t_start=-55, t_end=1500) == (-55, 1500)
        assert test_module.check_times(self.sim, t_start=None, t_end=1500) == (0, 1500)
        assert test_module.check_times(self.sim, t_start=None, t_end=None) == (0, 1000)

    def test_check_times_2(self):
        with pytest.raises(BluePyError):
            test_module.check_times(self.sim, t_start=1500, t_end=500)

    def test_check_times_3(self):
        res = test_module.check_times(self.sim, t_start=500,
                                      t_end=1500, report_name='soma.h5')
        npt.assert_almost_equal(res, [0, 10])
        res = test_module.check_times(self.sim, t_start=None,
                                      t_end=None, report_name='soma.h5')
        npt.assert_almost_equal(res, [0, 10])

    def test_get_figure(self):
        fig = test_module.get_figure()
        assert isinstance(fig, matplotlib.figure.Figure)
        npt.assert_allclose(fig.get_size_inches(), [10., 8.])

    def test_potential_axes_setup_method(self):
        fig = test_module.get_figure()
        ax = fig.add_subplot(111)
        test_module.potential_axes_update(ax, 'mean')
        assert ax.get_xlabel() == 'Time [ms]'
        assert ax.get_ylabel() == 'Avg volt. [mV]'
        test_module.potential_axes_update(ax, 'all', xlegend=False)
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == 'Voltage [mV]'
        test_module.potential_axes_update(ax, '')
        assert ax.get_ylabel() == ''

    @patch('bluepy.plotting.select_gids', return_value=[10, 20])
    def test_get_report_data_1(self, select_gids):
        tested = test_module.get_report_data(self.sim, 'soma.h5', [10, 20], 10, None, None, 1)
        assert set(tested) == {10, 20}
        assert set(tested.index) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_get_report_data_2(self):
        with pytest.raises(BluePyError):
            test_module.get_report_data(self.sim, 'soma.h5', [10, 20], 10, None, None, 1)

    @patch('bluepy.plotting.select_gids', return_value=[10, 20])
    def test_get_report_data_1_without_brion(self, select_gids):
        with PatchImport("brion"):
            tested = test_module.get_report_data(self.sim, 'soma.h5', [10, 20], 10, None, None, 1)
            assert set(tested) == {10, 20}
            assert set(tested.index) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
