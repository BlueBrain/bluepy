"""
Spike report access.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from cached_property import cached_property
from libsonata import SpikeReader

from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.impl.utils import _inc
from bluepy.utils import IDS_DTYPE


class SpikeReport:
    """Class to access spike reports"""

    def __init__(self, data):
        """
        Constructor. Use `SpikeReport.load` for creating instances of the class.

        Args:
            data (pd.Series): loaded data from either `_load_sonata` or `_load_dat`.
        """
        self._data = data

    @classmethod
    def load(cls, filepath):
        """
        Create an instance of report class depending on `filepath` file extension.

        Returns:
            instance of SpikeReport.
            Throws if file extension is unknown.
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        if ext == '.dat':
            return cls(cls._load_dat(filepath))
        elif ext == '.h5':
            return cls(cls._load_sonata(filepath))
        else:
            raise BluePyError("Unknown file extension of spikes report."
                              f" {filepath} must end with '.dat' or '.h5' extension.")

    @classmethod
    def _load_sonata(cls, filepath):
        """ Load spike reports in Sonata format. """
        reader = SpikeReader(filepath)
        names = reader.get_population_names()
        if len(names) > 1:
            raise BluePyError("Sonata spike reports with multiple populations are not supported")
        result = pd.DataFrame(reader[names[0]].get(), columns=['ids', 'times'])
        # pylint: disable=unsubscriptable-object
        result = result.set_index('times')['ids'].astype(IDS_DTYPE).sort_index()
        result[:] = _inc(result.values)
        return result

    @classmethod
    def _load_dat(cls, filepath):
        """ Load spike reports in .dat format. """
        result = pd.read_csv(
            filepath, skiprows=1, skipinitialspace=True, sep=r'\s+', names=['t', Cell.ID],
            dtype={'t': np.float64, Cell.ID: IDS_DTYPE}, index_col=['t']
        )
        result.sort_index(inplace=True)  # pylint: disable=maybe-no-member
        return result[Cell.ID]  # pylint: disable=unsubscriptable-object

    @cached_property
    def gids(self):
        """ GIDs of the measured cells. """
        return np.unique(self._data)

    def get(self, gids=None, t_start=None, t_end=None):
        """
        Fetch spikes from the report.

        If `gids` is provided, filter by GIDs.
        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            pandas Series with spiking GIDs indexed by sorted spike time.
        """
        result = self._data.loc[t_start:t_end]
        if gids is not None:
            result = result[result.isin(gids)]
        return result

    def get_gid(self, gid, t_start=None, t_end=None):
        """
        Fetch spikes from the report for a given `gid`.

        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            numpy array with sorted spike times.
        """
        return self._data[self._data == gid].loc[t_start:t_end].index.to_numpy()
