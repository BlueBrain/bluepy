""" Access to MVD2/MVD3 files. """
from collections.abc import Mapping, Sequence

import copy
import os

import numpy as np
import pandas as pd
from voxcell import CellCollection

from bluepy.exceptions import BluePyError, require
from bluepy.utils import ensure_list
from bluepy.utils.query import gids_by_filter


def loadMVD2(filepath):
    """ Read cell properties from MVD2.

        Returns a pandas.DataFrame with cell properties indexed by GIDs.
    """
    return CellCollection.load_mvd2(filepath).as_dataframe()


def loadMVD3(filepath):
    """ Read cell properties from MVD3.

        Returns a pandas.DataFrame with cell properties indexed by GIDs.
    """
    return CellCollection.load_mvd3(filepath).as_dataframe()


def load(filepath):
    """ Read cell properties from MVD2/MVD3.

        Returns a pandas.DataFrame with cell properties indexed by GIDs.
    """
    _, ext = os.path.splitext(filepath)
    if ext == ".mvd2":
        return loadMVD2(filepath)
    elif ext == ".mvd3":
        return loadMVD3(filepath)
    else:
        raise BluePyError(f"Unsupported file extension '{ext}'")


class MVDCellCollection:
    """ Access to cell properties stored in MVD2/3. """

    def __init__(self, mvd_path, targets=None):
        self._data = load(mvd_path)
        self._targets = targets

    @property
    def available_properties(self):
        """Set of available node properties."""
        return set(self._data.columns)

    def ids(self, group=None, limit=None, sample=None):
        """ GIDs corresponding to cell `group`. """
        # pylint: disable=too-many-branches
        def _split_props(group):
            """ Split dict-like group into 'meta' properties and all the rest. """
            props = copy.deepcopy(group)
            meta_props = {key[1:]: props.pop(key) for key in list(props) if key.startswith("$")}
            unknown_meta_props = set(meta_props) - set(['target'])
            if unknown_meta_props:
                raise BluePyError(f"Unknown meta props: {', '.join(unknown_meta_props)}")
            return props, meta_props

        preserve_order = False

        if group is None:
            result = self._data.index.values
        elif isinstance(group, str):
            require(self._targets is not None, "Targets not defined")
            result = self._targets.resolve(group)
        elif isinstance(group, Mapping):
            props, meta_props = _split_props(group)
            if 'target' in meta_props:
                require(self._targets is not None, "Targets not defined")
                target = self._targets.resolve(meta_props['target'])
                if len(target) == 0:
                    result = target
                elif np.min(target) > self._data.index.max():
                    require(not props, "Could not apply properties filter to external target")
                    result = target
                else:
                    result = gids_by_filter(self._data.loc[target], props=props)
            else:
                result = gids_by_filter(self._data, props=props)
        elif isinstance(group, np.ndarray):
            result = group
            preserve_order = True
        else:
            result = ensure_list(group)
            preserve_order = isinstance(group, Sequence)

        if sample is not None:
            if len(result) > 0:
                result = np.random.choice(result, sample, replace=False)
            preserve_order = False
        if limit is not None:
            result = result[:limit]

        result = np.array(result, dtype=int)
        if preserve_order:
            return result
        else:
            return np.unique(result)

    def get(self, group=None, properties=None):
        """ Cell properties as pandas Series / DataFrame. """
        if group is None:
            result = self._data
        elif isinstance(group, (int, np.integer)):
            self._check_id(group)
            result = self._data.loc[group]
        else:
            ids = self.ids(group)
            self._check_ids(ids)
            result = self._data.loc[ids]
        if properties is not None:
            result = result[properties]
        return result

    def _check_id(self, gid):
        """ Check that single GID belongs to the circuit. """
        if gid not in self._data.index:
            raise BluePyError(f"GID not found: {gid}")

    def _check_ids(self, gids):
        """ Check that GIDs belong to the circuit. """
        missing = pd.Index(gids).difference(self._data.index)
        if not missing.empty:
            raise BluePyError(f"GIDs not found: [{','.join(map(str, missing))}]")
