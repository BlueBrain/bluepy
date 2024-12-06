"""Compartment report access."""
import abc
from collections import namedtuple
import logging

import h5py
import numpy as np
import pandas as pd

from cached_property import cached_property
from libsonata import ElementReportReader

from bluepy.exceptions import require, BluePyError
from bluepy.enums import Cell, Section
from bluepy.impl.utils import _inc, _dec
from bluepy.utils import IDS_DTYPE

L = logging.getLogger(__name__)

START_TIME_KEY = 'start_time'
END_TIME_KEY = 'end_time'
TIME_STEP_KEY = 'time_step'

ReportData = namedtuple('ReportResult', ['data', 'times', 'ids'])


class ABCFrameReport(abc.ABC):
    """The abstract class/interface for the FrameReports."""

    def __init__(self, filepath):
        """The instance of the report."""
        self._reader = self._get_reader(filepath)

    @staticmethod
    @abc.abstractmethod
    def _get_reader(filepath):
        """The actual reader."""

    @cached_property
    @abc.abstractmethod
    def meta(self):
        """ Report meta data. """

    @property
    def t_start(self):
        """ Report start time. """
        return self.meta[START_TIME_KEY]

    @property
    def t_end(self):
        """ Report end time. """
        return self.meta[END_TIME_KEY]

    @property
    def t_step(self):
        """ Report time step. """
        return self.meta[TIME_STEP_KEY]

    @abc.abstractmethod
    def get(self, t_start=None, t_end=None, t_step=None, gids=None):
        """Collecting the data."""

    @abc.abstractmethod
    def gids(self):
        """Returns the gids present inside the report."""


class BrionReport(ABCFrameReport):
    """The brion report reader."""

    @staticmethod
    def _get_reader(filepath):
        """The actual reader."""
        import brion
        return brion.CompartmentReport(filepath)

    @cached_property
    def gids(self):
        """Returns the gids present inside the report."""
        return self._reader.gids

    @cached_property
    def meta(self):
        """ Report meta data. """
        return self._reader.metadata

    def get(self, t_start=None, t_end=None, t_step=None, gids=None):
        """Fetch data from the report.

        Args:
            t_start (float): Include only frames occurring at or after this time.
            t_end (float): Include only frames occurring before this time.
            t_step: time step (should be a multiple of report time step T; equals T by default)
            gids: GIDs of interest (no filtering by default)

        Returns:
            ReportData: ReportData containing all the data needed to create a dataframe (data, ids,
            times)
        """
        if gids is None:
            view = self._reader.create_view()
        else:
            # brion.CompartmentReport is not compliant with uint64 gids (raises error)
            gids = np.intersect1d(list(gids), self.gids).astype(IDS_DTYPE)
            if len(gids) == 0:
                return ReportData(data=[], times=[], ids=[])
            view = self._reader.create_view(gids)

        times, data = view.load(t_start, t_end, t_step)
        ids = np.array(list(map(tuple, view.mapping.index))).T
        return ReportData(data=data, times=times, ids=ids)


class LibSonataReader(ABCFrameReport):
    """The libsonata report reader."""
    _EPSILON = 0.1
    _DATASETS = {
        "TIME": "mapping/time",
        "INDEX_POINTER": "mapping/index_pointers",
        "ELEMENT_ID": "mapping/element_ids",
        "GIDS": "mapping/node_ids",
        "DATA": "data",
    }

    def __init__(self, filepath):
        # need to do the check_format before creating the libsonata reader or h5lib will throw a
        # tonne of error we cannot silent
        self._check_format(h5py.File(filepath, "r"))
        super().__init__(filepath)

    @classmethod
    def _check_format(cls, h5f):
        """ Check the hdf5 file format and return the report element """

        require('report' in h5f, "Missing 'report' key in H5 file")

        population_names = list(h5f['report'])
        require(len(population_names) == 1, "Only one population is supported.")

        population_name = population_names[0]
        report = h5f['report'][population_name]
        required_keys = cls._DATASETS.values()
        missing_keys = [key for key in required_keys if key not in report]
        require(
            not missing_keys,
            f"Missing keys in H5 '/report/{population_name}' [{', '.join(missing_keys)}]",
        )

        # check if gids and index_pointer can match
        gids_size = report['mapping/node_ids'].shape[0]
        indices_size = report['mapping/index_pointers'].shape[0]
        require(
            gids_size + 1 == indices_size,
            "The report mapping/index_pointers must have one element more than mapping/node_ids.",
        )

        # check if the last value from index_pointers is equal to the size of node_ids
        data_gids_count = report["data"].shape[1]
        last_pointer = report["mapping/index_pointers"][-1]
        require(
            data_gids_count == last_pointer,
            "mapping/index_pointers does not match with the data. Last index "
            f"pointer ({last_pointer}) should be equal to {data_gids_count}",
        )

        # check if all the data columns can be named
        element_ids_count = report["mapping/element_ids"].shape[0]
        require(
            data_gids_count == element_ids_count,
            "Missing element_ids values to match the data.",
        )

    @staticmethod
    def _get_reader(filepath):
        full_report = ElementReportReader(filepath)
        return full_report[list(full_report.get_population_names())[0]]

    @cached_property
    def meta(self):
        """ Report meta data. """
        times_properties = self._reader.times
        result = {
            START_TIME_KEY: times_properties[0],
            END_TIME_KEY: times_properties[1],
            TIME_STEP_KEY: times_properties[2],
        }
        return result

    def _excludes_up_value(self, t_end):
        """Shift, if needed, the upper time value.

        This function aims at mimicing Brion's time range behavior. That is, a rounding to
        the closest time step with upper exclusion.
        This function rounds t_end to the closest timestamp (same behavior as builtin round() but
        with float steps) and it excludes the value by shiting t_end by a small amount.

        Example of Brion behavior:
        - t_step = 0.1, t_end = 0.5 --> 0.5 is excluded
        - t_step = 0.1, t_end = 0.52 --> t_end is rounded to 0.5 and 0.5 is excluded
        - t_step = 0.1, t_end = 0.56 --> t_end is rounded to 0.6, 0.5 is kept but 0.6 is excluded
        """
        t_end = round(t_end / self.t_step) * self.t_step
        if np.isclose(t_end / self.t_step - round(t_end / self.t_step), 0):
            return t_end - self.t_step * self._EPSILON

        # Don t want to assert the above "if expression" due to possible pb in rounding with floats
        # the return below should never happens and if it can happen this is not a problem.
        return t_end  # pragma: no-cover

    def get(self, t_start=None, t_end=None, t_step=None, gids=None):
        """Fetch data from the report.

        Args:
            t_start (float): Include only frames occurring at or after this time.
            t_end (float): Include only frames occurring before this time.
            t_step: time step (should be a multiple of report time step T; equals T by default)
            gids: GIDs of interest, 1-based (no filtering by default)

        Returns:
            ReportData: ReportData containing all the data needed to create a dataframe (data, ids,
            times)
        """
        # If t_end is very close to a timestep it excludes this time step (compat with brion)
        t_end = self._excludes_up_value(t_end)
        if gids is None:
            gids = self.gids
        else:
            # filter out gids <= 0 because libsonata >= 0.1.7 does not accept negative gids
            gids = np.asarray(gids)
            gids = gids[(gids > 0).nonzero()]
        view = self._reader.get(node_ids=_dec(gids).tolist(),
                                tstart=t_start, tstop=t_end, tstride=round(t_step / self.t_step))

        if len(view.ids) == 0:
            return ReportData(data=[], times=[], ids=[])

        # keep the 1 based ids
        ids = np.asarray(view.ids, dtype=IDS_DTYPE).T
        ids[0, :] = _inc(ids[0, :])

        return ReportData(data=view.data, times=view.times, ids=ids)

    @cached_property
    def gids(self):
        """Returns the gids present inside the report."""
        return _inc(self._reader.get_node_ids())


class H5Reader(ABCFrameReport):
    """Implementation for reports in internal h5 format."""
    _DATASETS = {
        "TIME": "mapping/time",
        "INDEX_POINTER": "mapping/index_pointer",
        "ELEMENT_ID": "mapping/element_id",
        "GIDS": "mapping/gids",
        "DATA": "data",
    }

    def __init__(self, filepath):
        super().__init__(filepath)
        self._check_format(self._reader)

    @staticmethod
    def _get_reader(filepath):
        return h5py.File(filepath, "r")

    @cached_property
    def meta(self):
        """ Report meta data. """
        times_properties = self._get_dataset('TIME')[()]
        result = {
            START_TIME_KEY: times_properties[0],
            END_TIME_KEY: times_properties[1],
            TIME_STEP_KEY: times_properties[2],
        }
        return result

    @classmethod
    def _check_format(cls, h5f):
        """ Check the hdf5 file format """

        required_keys = cls._DATASETS.values()
        missing_keys = [key for key in required_keys if key not in h5f]
        require(
            not missing_keys,
            f"Missing keys in H5 file [{', '.join(missing_keys)}]",
        )

        # check if gids and index_pointer can match
        gids_size = h5f['mapping/gids'].shape[0]
        indices_size = h5f['mapping/index_pointer'].shape[0]
        require(
            gids_size == indices_size,
            "The report mapping/gids and mapping/index_pointer don't have the same size.",
        )

        # check if the last value from index_pointer included in data
        last_pointer = h5f["mapping/index_pointer"][-1]
        data_gids_count = h5f["data"].shape[1]
        require(
            data_gids_count >= last_pointer,
            "mapping/index_pointer does not match with the data. Last index "
            f"pointer ({last_pointer}) should be inferior than {data_gids_count}.",
        )

        # check if all the data columns can be named
        element_ids_count = h5f["mapping/element_id"].shape[0]
        require(
            data_gids_count == element_ids_count,
            "Missing element_ids values to match the data.",
        )

    def _get_dataset(self, key):
        """ Return the dataset for a given key. """
        return self._reader[self._DATASETS[key]]

    @cached_property
    def gids(self):
        """Returns the gids present inside the report."""
        gids = self._get_dataset('GIDS')[()]
        gids.sort()
        return gids

    def _find_frame(self, t):
        """ Find the corresponding frame for a given time. """
        require(self.t_start <= t <= self.t_end, "Out of time range")
        return np.round((t - self.t_start) / self.t_step).astype(int)

    def _get_offsets(self, gids, size):
        """Return a DataFrame of offsets for the specified gids."""
        _gids = self._get_dataset('GIDS')[()]
        _index_pointers = self._get_dataset('INDEX_POINTER')[()]
        _shifted_index_pointers = np.append(_index_pointers[1:], np.uint64(size))
        offsets = pd.DataFrame(
            data={
                "first_index": _index_pointers,
                "last_index": _shifted_index_pointers,
            },
            index=_gids,
        )
        return offsets.loc[gids]

    def _get_columns(self, offsets):
        """Return the array of tuples to be used as columns."""
        _element_ids = self._get_dataset("ELEMENT_ID")[()]
        return np.array(
            [
                (gid, _id)
                for gid in offsets.index
                for _id in _element_ids[
                    offsets.at[gid, "first_index"]:offsets.at[gid, "last_index"]
                ]
            ]
        ).T

    def _get_data(self, dataset, offsets, start, stop, step):
        """Return the selected data from the given dataset."""
        indices = np.concatenate(
            [
                np.arange(
                    offsets.at[gid, "first_index"],
                    offsets.at[gid, "last_index"],
                    dtype=np.uint64,
                )
                for gid in offsets.index
            ]
        )
        frames = np.arange(start, stop, step)
        return np.array(dataset[np.ix_(frames, indices)], dtype=np.float32)

    def get(self, t_start=None, t_end=None, t_step=None, gids=None):
        """Fetch data from the report.

        Args:
            t_start (float): Include only frames occurring at or after this time.
            t_end (float): Include only frames occurring before this time.
            t_step: time step (should be a multiple of report time step T; equals T by default)
            gids: GIDs of interest (no filtering by default)

        Returns:
            ReportData: ReportData containing all the data needed to create a dataframe (data, ids,
            times)
        """
        if gids is None:
            gids = self.gids
        else:
            gids = np.intersect1d(list(gids), self.gids).astype(IDS_DTYPE)

        if len(gids) == 0:
            return ReportData(data=[], times=[], ids=[])

        start = self._find_frame(t_start)
        stop = self._find_frame(t_end)
        step = int(np.round(t_step / self.t_step))

        # step is sanitized before
        assert (step >= 1) and np.isclose(step * self.t_step, t_step), \
            f"{t_step} is not a multiple of {self.t_step}"

        # convert to numpy array because very long lists (> 1000 elements)
        # may produce poor performance using h5py indexing
        dataset = self._get_dataset('DATA')[()]
        offsets = self._get_offsets(gids, size=dataset.shape[1])

        columns = self._get_columns(offsets)
        data = self._get_data(dataset, offsets, start, stop, step)
        times = self.t_start + self.t_step * np.arange(start, stop, step)[:data.shape[0]]
        return ReportData(data=data, times=times, ids=columns)


class TimeSeriesReport:
    """ Base class for Cell/Compartment/Synapse reports. """

    def __init__(self, filepath):
        self._reader = TimeSeriesReport._open(filepath)

    @staticmethod
    def _open(filepath):
        """Open the report file trying multiple readers.

        The lookup priority has been defined using the expected performances of each reader and
        their supported formats.

        Supported formats for each readers:
        - LibSonataReader can read sonata reports
        - BrionReport can read bbp, sonata and internal h5 reports
        - H5Reader can read internal h5 reports

        Performances:
        - LibSonataReader > BrionReport for sonata reports
        - BrionReport > H5Reader for h5 reports
        - BrionReport only one for the .bbp reports

        Hence the reader lookup priority : LibSonataReader --> BrionReport --> H5Reader.
        """
        try:
            res = LibSonataReader(filepath)
        except (BluePyError, RuntimeError, OSError):
            pass
        else:
            L.info("Using Libsonata to read the report.")
            return res
        try:
            res = BrionReport(filepath)
        except RuntimeError:
            pass  # pragma: no-cover
        except ImportError:
            from pathlib import Path
            if Path(filepath).suffix.lower() == ".bbp":
                raise BluePyError("Brion is mandatory for the '.bbp' reports but is not installed."
                                  "Please use: pip install brion")
        else:
            L.info("Using Brion to read the report.")
            return res

        try:
            res = H5Reader(filepath)
        except BluePyError:
            raise BluePyError("Cannot read this report with bluepy. "
                              "Please, double check your reports.")

        L.info("Using internal h5 reader to read the report.")
        return res

    @property
    def gids(self):
        """Returns the gids present inside the report."""
        return self._reader.gids

    @property
    def meta(self):
        """ Report meta data. """
        return self._reader.meta

    @property
    def t_start(self):
        """ Report start time. """
        return self._reader.t_start

    @property
    def t_end(self):
        """ Report end time. """
        return self._reader.t_end

    @property
    def t_step(self):
        """ Report time step. """
        return self._reader.t_step

    def _sanitize_time_requests(self, t_start, t_end, t_step):
        if t_start is None:
            t_start = self.t_start
        else:
            require(self.t_start <= t_start <= self.t_end, "t_start is out of time bounds")

        if t_end is None:
            t_end = self.t_end
        else:
            require(self.t_start <= t_end <= self.t_end, "t_end is out of time bounds")

        if t_step is None:
            t_step = self.t_step
        return t_start, t_end, t_step

    def get(self, t_start=None, t_end=None, t_step=None, gids=None):
        """
        Fetch data from the report.

        Args:
            t_start (float): Include only frames occurring at or after this time.
            t_end (float): Include only frames occurring before this time.
            t_step: time step (should be a multiple of report time step T; equals T by default)
            gids: GIDs of interest (no filtering by default)

        Returns:
            A (multi)indexed DataFrame with measurements,
            where rows correspond to time slices, and columns to compartments.

        See also:
            https://developer.humanbrainproject.eu/docs/bluepy/latest/tutorial.html#somareport
        """
        t_start, t_end, t_step = self._sanitize_time_requests(t_start, t_end, t_step)
        val = t_step / self.t_step
        if not np.isclose(val, round(val)):
            raise BluePyError(f"{t_step} is not a multiple of {self.t_step}")
        report_res = self._reader.get(t_start=t_start, t_end=t_end, t_step=t_step, gids=gids)
        if len(report_res.ids) == 0:
            return pd.DataFrame()
        return pd.DataFrame(data=report_res.data,
                            index=pd.Index(report_res.times, name='time'),
                            columns=self._wrap_index(report_res.ids))

    def get_gid(self, gid, t_start=None, t_end=None, t_step=None):
        """
        Fetch data from the report for a given `gid`.

        Args:
            gid: GID of interest
            t_start (float): Include only frames occurring at or after this time.
            t_end (float): Include only frames occurring before this time.
            t_step: time step (should be a multiple of report time step T; equals T by default)

        Returns:
            pandas Series/DataFrame with measurements indexed by time slices.

        See also:
            https://developer.humanbrainproject.eu/docs/bluepy/latest/tutorial.html#somareport
        """
        if gid not in self.gids:
            raise BluePyError(f"No such GID in the report: {gid}")
        return self.get(gids=[gid], t_start=t_start, t_end=t_end, t_step=t_step)

    @staticmethod
    def _wrap_index(index):
        """ Create pandas (Multi)Index from 'raw' index. """
        index = np.asarray(index, dtype=IDS_DTYPE)
        return pd.MultiIndex.from_arrays(index)


class SomaReport(TimeSeriesReport):
    """ Report object for Cell reports. """

    @staticmethod
    def _wrap_index(index):
        index = np.asarray(index, dtype=IDS_DTYPE)
        return pd.Index(index[0, :], name=Cell.ID)

    def get_gid(self, gid, t_start=None, t_end=None, t_step=None):
        return super().get_gid(gid, t_start=t_start, t_end=t_end, t_step=t_step)[gid]


class CompartmentReport(TimeSeriesReport):
    """ Report object for Compartment reports. """

    @staticmethod
    def _wrap_index(index):
        index = np.asarray(index, dtype=IDS_DTYPE)
        return pd.MultiIndex.from_arrays(index, names=[Cell.ID, Section.ID])

    def get_gid(self, gid, t_start=None, t_end=None, t_step=None):
        return super().get_gid(gid, t_start=t_start, t_end=t_end, t_step=t_step)[gid]


class SynapseReport(TimeSeriesReport):
    """ Report object for Synapse reports. """

    @staticmethod
    def _wrap_index(index):
        index = np.asarray(index, dtype=IDS_DTYPE)
        return pd.MultiIndex.from_arrays(index)

    def get_synapses(self, synapse_ids, t_start=None, t_end=None, t_step=None):
        """ Fetch data from the report for selected synapse IDs. """
        gids = set(id_[0] for id_ in synapse_ids)
        result = self.get(gids=gids, t_start=t_start, t_end=t_end, t_step=t_step)
        return result[synapse_ids]
