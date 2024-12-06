import os.path
import types
from unittest.mock import Mock, patch

import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from utils import copy_file

import bluepy.impl.compartment_report as test_module
from bluepy.enums import Cell, Section
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

TEST_OLD_REPORT_H5 = os.path.join(TEST_DATA_DIR, "soma_v1.h5")
TEST_REPORT_H5 = os.path.join(TEST_DATA_DIR, "soma.h5")
TEST_REPORT_SONATA = os.path.join(TEST_DATA_DIR, "soma_sonata.h5")
TEST_COMPARTMENT_REPORT_H5 = os.path.join(TEST_DATA_DIR, "AllCompartments.filtered.internal.h5")
TEST_COMPARTMENT_REPORT_SONATA = os.path.join(TEST_DATA_DIR, "AllCompartments.filtered.sonata.h5")


def _patch_reader(cls, reader):
    with patch.object(test_module.TimeSeriesReport, '_open', return_value=reader):
        return cls(None)


def _assert_report_data_not_empty(report_data):
    assert isinstance(report_data, test_module.ReportData)
    assert len(report_data.ids) > 0
    assert len(report_data.times) > 0
    assert len(report_data.data) > 0


def _test_sonata_compatibility(reader_sonata, reader_h5, t_start, t_end, t_step, gids):
    metadata_sonata = reader_sonata.meta
    gids_sonata = reader_sonata.gids
    data_sonata = reader_sonata._reader.get(t_start, t_end, t_step, gids)

    metadata_new = reader_h5.meta
    metadata_new = {x: metadata_new[x] for x in metadata_new if x in list(metadata_sonata)}
    gids_new = reader_h5.gids
    data_new = reader_h5._reader.get(t_start, t_end, t_step, gids)

    _assert_report_data_not_empty(data_sonata)
    _assert_report_data_not_empty(data_new)

    npt.assert_allclose(gids_sonata, gids_new)
    assert metadata_sonata == metadata_new
    npt.assert_allclose(data_sonata.data, data_new.data)
    npt.assert_allclose(data_sonata.times, data_new.times)
    npt.assert_allclose(data_sonata.ids, data_new.ids)


def test_brion_is_installed():
    import brion
    assert isinstance(brion, types.ModuleType)


class TestCompareReaders:
    """This test is run with brion as h5 backend if brion is installed and it is run with
    the h5 "bluepy backend" if brion is not installed.
    """
    def setup_method(self):
        self._reader_h5 = test_module.TimeSeriesReport(TEST_REPORT_H5)
        self._reader_sonata = test_module.TimeSeriesReport(TEST_REPORT_SONATA)

    def test_impl(self):
        # BrionReport is used when brion is available
        assert isinstance(self._reader_h5._reader, test_module.BrionReport)
        assert isinstance(self._reader_sonata._reader, test_module.LibSonataReader)

    def test_check_format_old_in_new(self):
        with pytest.raises(BluePyError, match="Missing keys in H5 file"):
            test_module.H5Reader(TEST_OLD_REPORT_H5)

    def test_check_format_old_in_sonata(self):
        with pytest.raises(BluePyError, match="Missing 'report' key in H5 file"):
            test_module.LibSonataReader(TEST_OLD_REPORT_H5)

    def test_check_format_new_in_sonata(self):
        with pytest.raises(BluePyError, match="Missing 'report' key in H5 file"):
            test_module.LibSonataReader(TEST_REPORT_H5)

    def test_new_sonata_compatibility_simple(self):
        _test_sonata_compatibility(
            reader_sonata=self._reader_sonata,
            reader_h5=self._reader_h5,
            t_start=0,
            t_end=1,
            t_step=0.1,
            gids=[550, 560, 570, 580, 590, 600, 781, 811],
        )

    def test_new_sonata_compatibility(self):
        _test_sonata_compatibility(
            reader_sonata=self._reader_sonata,
            reader_h5=self._reader_h5,
            t_start=2,
            t_end=10,
            t_step=1,
            gids=[550, 560, 570, 580, 590, 600, 781, 811],
        )

    def test_bad_reports(self):
        # missing root elements
        for value in ['mapping', 'data']:
            with copy_file(TEST_REPORT_H5) as filepath:
                with h5py.File(filepath, "r+") as h5:
                    del h5[value]
                with pytest.raises(BluePyError, match="Missing keys in H5 file"):
                    test_module.H5Reader(filepath)

        # missing field in mapping
        for value in ['gids', 'element_id', 'index_pointer', 'time']:
            with copy_file(TEST_REPORT_H5) as filepath:
                with h5py.File(filepath, "r+") as h5:
                    del h5["/mapping/" + value]
                with pytest.raises(BluePyError, match="Missing keys in H5 file"):
                    test_module.H5Reader(filepath)

        # pointer index values not compatible with gids size
        with copy_file(TEST_REPORT_H5) as filepath:
            with h5py.File(filepath, "r+") as h5:
                size = h5["/mapping/index_pointer"][()].shape[0]
                del h5["/mapping/index_pointer"]
                h5.create_dataset("/mapping/index_pointer", data=np.arange(size + 1))
            match = "The report mapping/gids and mapping/index_pointer don't have the same size"
            with pytest.raises(BluePyError, match=match):
                test_module.H5Reader(filepath)

        # index pointer value cannot match data size
        with copy_file(TEST_REPORT_H5) as filepath:
            with h5py.File(filepath, "r+") as h5:
                shape = list(h5["data"].shape)
                shape[1] -= 5
                del h5["data"]
                h5.create_dataset("data", data=np.random.random(shape))
            match = "mapping/index_pointer does not match with the data. Last index pointer"
            with pytest.raises(BluePyError, match=match):
                test_module.H5Reader(filepath)

        # element_id size does not match data size
        with copy_file(TEST_REPORT_H5) as filepath:
            with h5py.File(filepath, "r+") as h5:
                size = h5["/mapping/element_id"][()].shape[0] + 1
                del h5["/mapping/element_id"]
                h5.create_dataset("/mapping/element_id", data=np.arange(size))
            with pytest.raises(BluePyError, match="Missing element_ids values to match the data"):
                test_module.H5Reader(filepath)

    def test_bad_reports_sonata(self):
        prefix = "/report/All/"

        # missing root elements
        for value in ['report', prefix, prefix + 'mapping', prefix + 'data']:
            with copy_file(TEST_REPORT_SONATA) as filepath:
                with h5py.File(filepath, "r+") as h5:
                    del h5[value]
                with pytest.raises(BluePyError):
                    test_module.LibSonataReader(filepath)

        # missing field in mapping
        for value in ['node_ids', 'element_ids', 'index_pointers', 'time']:
            with copy_file(TEST_REPORT_SONATA) as filepath:
                with h5py.File(filepath, "r+") as h5:
                    del h5[prefix + "mapping/" + value]
                with pytest.raises(BluePyError):
                    test_module.LibSonataReader(filepath)

        # pointer index size not compatible with gids size
        with copy_file(TEST_REPORT_SONATA) as filepath:
            with h5py.File(filepath, "r+") as h5:
                dataset_key = prefix + "mapping/index_pointers"
                size = h5[dataset_key][()].shape[0]
                del h5[dataset_key]
                h5.create_dataset(dataset_key, data=np.arange(size + 1))
            match = (
                "The report mapping/index_pointers must have one element more than mapping/node_ids"
            )
            with pytest.raises(BluePyError, match=match):
                test_module.LibSonataReader(filepath)

        # pointer index value not compatible with gids size
        with copy_file(TEST_REPORT_SONATA) as filepath:
            with h5py.File(filepath, "r+") as h5:
                dataset_key = prefix + "mapping/index_pointers"
                h5[dataset_key][-1] = h5[dataset_key][-1] + 1
            match = "mapping/index_pointers does not match with the data. Last index pointer"
            with pytest.raises(BluePyError, match=match):
                test_module.LibSonataReader(filepath)

        # index pointer value cannot match data size
        with copy_file(TEST_REPORT_SONATA) as filepath:
            with h5py.File(filepath, "r+") as h5:
                dataset_key = prefix + "data"
                shape = list(h5[dataset_key].shape)
                shape[1] -= 5
                del h5[dataset_key]
                h5.create_dataset(dataset_key, data=np.random.random(shape))
            match = "mapping/index_pointers does not match with the data. Last index pointer"
            with pytest.raises(BluePyError, match=match):
                test_module.LibSonataReader(filepath)

        # element_id size does not match data size
        with copy_file(TEST_REPORT_SONATA) as filepath:
            with h5py.File(filepath, "r+") as h5:
                dataset_key = prefix + "mapping/element_ids"
                size = h5[dataset_key][()].shape[0] + 1
                del h5[dataset_key]
                h5.create_dataset(dataset_key, data=np.arange(size))
            with pytest.raises(BluePyError, match="Missing element_ids values to match the data"):
                test_module.LibSonataReader(filepath)


class TestCompareReadersWithCompartmentReport:
    """This test is run with brion as h5 backend if brion is installed and it is run with
    the h5 "bluepy backend" if brion is not installed.
    """
    def setup_method(self):
        self._reader_h5 = test_module.TimeSeriesReport(TEST_COMPARTMENT_REPORT_H5)
        self._reader_sonata = test_module.TimeSeriesReport(TEST_COMPARTMENT_REPORT_SONATA)

    def test_impl(self):
        # BrionReport is used when brion is available
        assert isinstance(self._reader_h5._reader, test_module.BrionReport)
        assert isinstance(self._reader_sonata._reader, test_module.LibSonataReader)

    def test_sonata_compatibility(self):
        _test_sonata_compatibility(
            reader_sonata=self._reader_sonata,
            reader_h5=self._reader_h5,
            t_start=2,
            t_end=10,
            t_step=1,
            gids=[3, 5, 9999999],
        )


class TestTimeSeriesReport:
    """This test is run with brion as h5 backend if brion is installed and it is run with
    the h5 "bluepy backend" if brion is not installed.
    """
    def setup_method(self):
        self._report = test_module.TimeSeriesReport(TEST_REPORT_H5)

    def test_meta(self):
        assert self._report.t_start == 0.0
        assert self._report.t_end == 10.0
        assert self._report.t_step == 0.1

    def test_gids(self):
        assert len(self._report.gids) == 50
        npt.assert_equal(self._report.gids[:5], [10, 20, 30, 40, 50])

    def test_get_no_args(self):
        result = self._report.get()
        assert result.shape == (100, 50)
        assert result.columns.values[:2].tolist() == [(10, 0), (20, 0)]

    def test_get_time_range_t_end_on_timestamp(self):
        result = self._report.get(t_start=0.0, t_end=0.3)
        assert result.shape == (3, 50)
        expected = [
            [-65.04298, -65.03759],
            [-65.05032, -65.04214],
            [-65.07193, -65.04762],
        ]
        npt.assert_almost_equal(result.values[:, :2], expected, decimal=5)

    def test_get_time_range_t_end_rounding_down(self):
        result = self._report.get(t_start=0.0, t_end=0.34)
        assert result.shape == (3, 50)
        expected = [
            [-65.04298, -65.03759],
            [-65.05032, -65.04214],
            [-65.07193, -65.04762],
        ]
        npt.assert_almost_equal(result.values[:, :2], expected, decimal=5)

    def test_get_time_range_t_end_rounding_middle(self):
        result = self._report.get(t_start=0.0, t_end=0.35)
        assert result.shape == (3, 50)
        expected = [
            [-65.04298, -65.03759],
            [-65.05032, -65.04214],
            [-65.07193, -65.04762],
        ]
        npt.assert_almost_equal(result.values[:, :2], expected, decimal=5)

    def test_get_time_range_t_end_rounding_up(self):
        result = self._report.get(t_start=0.0, t_end=0.36)
        assert result.shape == (4, 50)
        expected = [
            [-65.04298, -65.03759],
            [-65.05032, -65.04214],
            [-65.07193, -65.04762],
            [-65.08295, -65.035]
        ]
        npt.assert_almost_equal(result.values[:, :2], expected, decimal=5)

    def test_get_full_time_range(self):
        t_start, t_end = self._report.t_start, self._report.t_end
        # should not throw
        _ = self._report.get(t_start=t_start, t_end=t_end, gids=[20], t_step=10)

    def test_get_time_range_downsample(self):
        result = self._report.get(t_start=0, t_end=0.3, t_step=0.2)
        assert result.shape == (2, 50)
        expected = [
            [-65.04298, -65.03759],
            [-65.07193, -65.04762],
        ]
        npt.assert_almost_equal(result.values[:, :2], expected, decimal=5)

    def test_get_gid_filtering(self):
        result = self._report.get(t_start=0, t_end=0.2, gids=[20])
        assert result.columns.values.tolist() == [(20, 0)]
        expected = [
            [-65.03759],
            [-65.04214],
        ]
        npt.assert_almost_equal(result.values, expected, decimal=5)

    def test_get_gid_bad_values(self):
        # negative gids are invalid
        result = self._report.get(t_start=0, t_end=0.2, gids=[-1])
        assert result.empty
        # gid 0 is invalid too, because it's 1-based and it will be decreased by 1
        result = self._report.get(t_start=0, t_end=0.2, gids=[0])
        assert result.empty
        result = self._report.get(t_start=0, t_end=0.2, gids=[999999])
        assert result.empty
        result = self._report.get(t_start=0, t_end=0.2, gids=[-1, 999999])
        assert result.empty
        result = self._report.get(t_start=0, t_end=0.2, gids=[-1, 20, 999999])
        assert result.columns.values.tolist() == [(20, 0)]
        expected = [
            [-65.03759],
            [-65.04214],
        ]
        npt.assert_almost_equal(result.values, expected, decimal=5)

    def test_get_gid_filtering_empty(self):
        result = self._report.get(gids=[])
        assert result.empty

    def test_get_raises(self):
        pytest.raises(BluePyError, self._report.get, t_start=-1)
        pytest.raises(BluePyError, self._report.get, t_end=101)
        pytest.raises(BluePyError, self._report.get, t_step=0.11)

    def test_get_gid(self):
        actual = self._report.get_gid(10, t_start=0.0, t_end=0.2)
        expected = [
            [-65.042976],
            [-65.050315],
        ]
        assert actual.shape == (2, 1)
        npt.assert_almost_equal(actual.values, expected, decimal=5)

    def test_get_gid_raises(self):
        pytest.raises(BluePyError, self._report.get_gid, 9999)
        pytest.raises(BluePyError, self._report.get_gid, -1)

    def test_column_index(self):
        actual = self._report.get(gids=[10, 20, 9999])
        pdt.assert_index_equal(
            actual.columns,
            pd.MultiIndex.from_tuples([(10, 0), (20, 0)]),
        )


class TestTimeSeriesReportSonata(TestTimeSeriesReport):
    """The sonata file is a strict conversion from the h5 to the sonata version so we can use the
    same tests and the same asserts used in TestTimeSeriesReport and just change the input file."""
    def setup_method(self):
        self._report = test_module.TimeSeriesReport(TEST_REPORT_SONATA)

    def test_impl(self):
        # the reader implementation should be the same, whether or not brion is installed
        assert isinstance(self._report._reader, test_module.LibSonataReader)


class TestTimeSeriesReportInternalH5(TestTimeSeriesReport):
    def setup_method(self):
        with patch.object(test_module.BrionReport, '_get_reader', side_effect=ImportError):
            self._report = test_module.TimeSeriesReport(TEST_REPORT_H5)


class TestSomaReport:
    def setup_method(self):
        reader = Mock()
        reader.meta = {'start_time': 0.0, 'end_time': 1.0, 'time_step': 0.1}
        reader.t_step = reader.meta['time_step']
        reader.t_end = reader.meta['end_time']
        reader.t_start = reader.meta['start_time']
        reader.gids = [1, 2, 3]

        reader.get.return_value = test_module.ReportData(
            data=None,
            times=[0.0, 0.1],
            # mock the ids with dtype uint64 as returned by libsonata >= 0.1.10
            ids=np.asarray([[1, 0], [2, 0]], dtype=np.uint64).T,
        )
        self._report = _patch_reader(test_module.SomaReport, reader)

    def test_column_index(self):
        actual = self._report.get(gids=[1, 2, 9999])
        pdt.assert_index_equal(actual.columns, pd.Index([1, 2], name=Cell.ID))

    def test_get_gid(self):
        result = self._report.get_gid(1)
        assert result.name == 1


class TestCompartmentReport:
    def setup_method(self):
        reader = Mock()
        reader.meta = {'start_time': 0.0, 'end_time': 1.0, 'time_step': 0.1}
        reader.gids = [1, 2, 3]
        reader.t_step = reader.meta['time_step']
        reader.t_end = reader.meta['end_time']
        reader.t_start = reader.meta['start_time']
        reader.get.return_value = test_module.ReportData(
            data=None,
            times=[0.0, 0.1],
            # mock the ids with dtype uint64 as returned by libsonata >= 0.1.10
            ids=np.asarray([[1, 0], [1, 1], [2, 0]], dtype=np.uint64).T,
        )
        self._report = _patch_reader(test_module.CompartmentReport, reader)

    def test_column_index(self):
        actual = self._report.get(gids=[1, 2, 9999])
        pdt.assert_index_equal(actual.columns,
                               pd.MultiIndex.from_tuples([(1, 0), (1, 1), (2, 0)],
                                                         names=[Cell.ID, Section.ID]))

    def test_get_gid(self):
        result = self._report.get_gid(1)
        pdt.assert_index_equal(result.columns, pd.Index([0, 1], name=Section.ID))


class TestSynapseReport:
    def setup_method(self):
        reader = Mock()
        reader.meta = {'start_time': 0.0, 'end_time': 1.0, 'time_step': 0.1}
        reader.gids = [1, 2, 3]
        reader.t_step = reader.meta['time_step']
        reader.t_end = reader.meta['end_time']
        reader.t_start = reader.meta['start_time']
        self._synapse_ids = np.array([[1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]).T
        reader.get.return_value = test_module.ReportData(
            data=[[11.0, 11.1, 11.2, 12.0, 12.1],
                  [21.0, 21.1, 21.2, 22.0, 22.1]],
            times=[0.0, 0.1],
            # mock the ids with dtype uint64 as returned by libsonata >= 0.1.10
            ids=np.asarray(self._synapse_ids, dtype=np.uint64),
        )
        self._report = _patch_reader(test_module.SynapseReport, reader)

    def test_get_gid(self):
        result = self._report.get_gid(2)
        pdt.assert_index_equal(result.columns, pd.MultiIndex.from_arrays(self._synapse_ids))

    def test_get_synapses(self):
        synapse_ids = [(1, 1), (1, 2), (2, 1)]
        actual = self._report.get_synapses(synapse_ids)
        expected = pd.DataFrame([
            [11.1, 11.2, 12.1],
            [21.1, 21.2, 22.1],
        ],
            index=pd.Index([0.0, 0.1], name='time'),
            columns=pd.Index(synapse_ids)
        )
        pdt.assert_frame_equal(actual, expected)

    def test_get_synapses_missing(self):
        pytest.raises(KeyError, self._report.get_synapses, [(10, -1)])
        pytest.raises(KeyError, self._report.get_synapses, [(-1, 0)])


def test_bbp_reports_with_brion():
    # I cannot convert sonata/h5 to bbp so I consider the minimal testing here for this format
    filepath = os.path.join(TEST_DATA_DIR, "soma.bbp")
    assert isinstance(test_module.TimeSeriesReport(filepath)._reader, test_module.BrionReport)


def test_full_fails_report():
    # this report will fail for all readers
    with copy_file(TEST_REPORT_H5) as filepath:
        with h5py.File(filepath, "r+") as h5:
            del h5['data']
        with pytest.raises(BluePyError, match="Cannot read this report with bluepy"):
            test_module.TimeSeriesReport(filepath)
