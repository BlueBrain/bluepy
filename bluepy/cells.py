""" Access to cell properties. """
import os
import logging

import pandas as pd
from cached_property import cached_property

from bluepy.exceptions import require
from bluepy.utils.url import get_file_path_by_url
from bluepy.enums import Cell
from bluepy.impl.target import TargetContext
from bluepy.exceptions import BluePyError

L = logging.getLogger(__name__)


class CellCollection:
    """ Access to cell properties. """
    def __init__(self, url, targets=None, spatial_index=None):
        self._targets = TargetContext.load(map(get_file_path_by_url, targets or []))
        filepath = get_file_path_by_url(url)
        if filepath.endswith(".sonata") or filepath.endswith(".h5"):
            from bluepy.impl.cells_sonata import SonataCellCollection
            self._impl = SonataCellCollection(filepath, self._targets)
            self._targets.filter(self._impl.node_sets, inplace=True)
        elif filepath.endswith(".mvd2") or filepath.endswith(".mvd3"):
            from bluepy.impl.cells_mvd import MVDCellCollection
            self._impl = MVDCellCollection(get_file_path_by_url(url), self._targets)
        else:
            raise BluePyError(f"{filepath} cannot be opened in bluepy. "
                              "Use .mvd2/.mvd3/.sonata/.h5 file format")
        self._index_url = spatial_index

    @property
    def available_properties(self):
        """Set of available cell properties."""
        return self._impl.available_properties

    def ids(self, group=None, limit=None, sample=None):
        """ GIDs corresponding to cell `group`.

            `group` could be:
                - single GID (int)
                - list of GIDs (list-like)
                - string (target name)
                - properties filter (dict-like)
                - None (i.e. 'all')

            If `sample` is specified, `sample` GIDs are randomly chosen from the match result.
            If `limit` is specified, first `limit` GIDs from the match result are returned.

            To combine selection by target name with properties filter,
            pass '$target': <target name> as one of properties filter keys.

            If `group` is a sequence (list or array), its order is preserved.
            Otherwise return result is sorted and contains no duplicates.
        """
        return self._impl.ids(group=group, limit=limit, sample=sample)

    def get(self, group=None, properties=None):
        """ Cell properties as pandas Series / DataFrame.

            `group` could be:
                - single GID (int)
                - list of GIDs (list-like)
                - string (target name)
                - properties filter (dict-like)
                - None (i.e. 'all')

            If `properties` is specified, return only selected properties (all by default).

            Returns:
                pandas Series if single GID is passed as `group`.
                pandas DataFrame indexed by GIDs otherwise.
        """
        return self._impl.get(group=group, properties=properties)

    def positions(self, group=None):
        """
        Cell position(s) as pandas Series / DataFrame.

        Returns:
            pandas ('x', 'y', 'z') Series if single GID is passed as `group`.
            pandas ('x', 'y', 'z') DataFrame indexed by GIDs otherwise
        """
        props = {
            Cell.X: 'x',
            Cell.Y: 'y',
            Cell.Z: 'z',
        }
        result = self._impl.get(group=group, properties=list(props))
        return result.rename(props).astype(float)[['x', 'y', 'z']]

    def orientations(self, group=None):
        """
        Cell orientation(s) as pandas Series / DataFrame.

        Returns:
            3x3 rotation matrix if single GID is passed as `group`.
            pandas Series with rotation matrices indexed by GIDs otherwise
        """
        return self._impl.get(group=group, properties=Cell.ORIENTATION)

    def count(self, group=None):
        """ Total number of cells for a given cell group. """
        return len(self.ids(group))

    @property
    def mtypes(self):
        """ Set of cell mtypes present in cell collection. """
        res = self.get(properties=Cell.MTYPE)
        if isinstance(res, pd.CategoricalDtype):
            return set(res.cat.categories)
        return set(res.unique())

    @property
    def etypes(self):
        """ Set of cell etypes present in cell collection. """
        res = self.get(properties=Cell.ETYPE)
        if isinstance(res, pd.CategoricalDtype):
            return set(res.cat.categories)
        return set(res.unique())

    @cached_property
    def targets(self):
        """ Set of available target names. """
        return set(self._targets.names)

    @cached_property
    def spatial_index(self):
        """ Spatial index. """
        require(self._index_url is not None, "Spatial index not defined")
        from bluepy.index.indices import SomaIndex
        return SomaIndex(os.path.join(get_file_path_by_url(self._index_url), "SOMA"))
