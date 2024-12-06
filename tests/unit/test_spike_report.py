import abc
import os.path

import numpy as np
import numpy.testing as npt
import pytest

import bluepy.impl.spike_report as test_module
from bluepy.exceptions import BluePyError
from bluepy.utils import IDS_DTYPE

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def check_times_close(actual, expected):
    npt.assert_allclose(actual, expected, atol=1e-6, rtol=0.0)


def check_gids_equal(actual, expected):
    assert np.asarray(expected).dtype == IDS_DTYPE
    assert np.asarray(actual).dtype == IDS_DTYPE
    npt.assert_equal(actual, expected)


def test_load_unknown_file():
    with pytest.raises(BluePyError) as e:
        test_module.SpikeReport.load('unknow_file_extension')
    assert 'Unknown file extension of spikes report' in str(e.value)


class TestSpikeReport(abc.ABC):

    def test_gids(self):
        check_gids_equal(self._report.gids, [11, 22, 33])

    def test_get(self):
        result = self._report.get()
        check_times_close(result.index, [0.000001, 13.0, 13.15, 18.5, 19.0, 10000.000001])
        check_gids_equal(result.values, [22, 33, 11, 11, 33, 22])

    def test_get_time_range(self):
        result = self._report.get(t_start=15, t_end=20)
        check_times_close(result.index, [18.5, 19.0])
        check_gids_equal(result.values, [11, 33])

    def test_get_gids(self):
        result = self._report.get(gids=[33])
        check_times_close(result.index, [13.0, 19.0])
        check_gids_equal(result.values, [33, 33])

    def test_get_gid(self):
        result = self._report.get_gid(33)
        check_times_close(result, [13.0, 19.0])

    def test_get_gid_time_range(self):
        result = self._report.get_gid(33, t_start=0, t_end=15)
        check_times_close(result, [13.0])

    def test_get_gid_no_spikes(self):
        result = self._report.get_gid(99)
        check_times_close(result, [])


class TestSpikeReportDat(TestSpikeReport):

    def setup_method(self):
        self._report = test_module.SpikeReport.load(os.path.join(TEST_DATA_DIR, "out.dat"))


class TestSpikeReportSonata(TestSpikeReport):

    def setup_method(self):
        self._report = test_module.SpikeReport.load(os.path.join(TEST_DATA_DIR, "spikes.h5"))


# delete this class so it won't be tested
del TestSpikeReport
