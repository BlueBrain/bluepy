""" Access to nrn*.h5 files. """

from builtins import map  # pylint: disable=redefined-builtin

import os
import h5py
import numpy as np
import pandas as pd
from morphio import SectionType

from cached_property import cached_property

from bluepy.enums import Cell, Direction, Section, Segment, Synapse
from bluepy.exceptions import BluePyError
from bluepy.impl.utils import _calc_morph_features
from bluepy.utils import ensure_list, gid2str, normalize_endianness, group_by_first

NEURITE_TYPE_MAP = {
    0: SectionType.soma,
    1: SectionType.axon,
    2: SectionType.basal_dendrite,
    3: SectionType.apical_dendrite,
}

NRN_PROPERTIES = {
    Synapse.PRE_GID: (0, int),
    Synapse.POST_GID: (0, int),
    Synapse.AXONAL_DELAY: (1, float),
    Synapse.POST_SECTION_ID: (2, int),
    Synapse.POST_SEGMENT_ID: (3, int),
    Synapse.POST_SEGMENT_OFFSET: (4, float),
    Synapse.PRE_SECTION_ID: (5, int),
    Synapse.PRE_SEGMENT_ID: (6, int),
    Synapse.PRE_SEGMENT_OFFSET: (7, float),
    Synapse.G_SYNX: (8, float),
    Synapse.U_SYN: (9, float),
    Synapse.D_SYN: (10, float),
    Synapse.F_SYN: (11, float),
    Synapse.DTC: (12, float),
    Synapse.TYPE: (13, int),
    Synapse.POST_BRANCH_ORDER: (15, lambda x: int(x) - 1),
    Synapse.PRE_BRANCH_ORDER: (16, lambda x: int(x) - 1),
    Synapse.NRRP: (17, float),
    Synapse.POST_BRANCH_TYPE: (18, lambda x: NEURITE_TYPE_MAP[int(x)]),
}

NRN_POSITIONS = {
    Synapse.PRE_X_CONTOUR: (1, float),
    Synapse.PRE_Y_CONTOUR: (2, float),
    Synapse.PRE_Z_CONTOUR: (3, float),
    Synapse.POST_X_CONTOUR: (4, float),
    Synapse.POST_Y_CONTOUR: (5, float),
    Synapse.POST_Z_CONTOUR: (6, float),
    Synapse.PRE_X_CENTER: (7, float),
    Synapse.PRE_Y_CENTER: (8, float),
    Synapse.PRE_Z_CENTER: (9, float),
    Synapse.POST_X_CENTER: (10, float),
    Synapse.POST_Y_CENTER: (11, float),
    Synapse.POST_Z_CENTER: (12, float),
}


def _choose_optimal_layout_direction(pre_gids, post_gids):
    """ Choose between afferent / efferent layout for finding connections. """
    if post_gids is None and pre_gids is None:
        raise BluePyError("Either `pre` or `post` should be specified")

    if pre_gids is None:
        result = Direction.AFFERENT
    elif post_gids is None:
        result = Direction.EFFERENT
    elif len(pre_gids) < len(post_gids):
        result = Direction.EFFERENT
    else:
        result = Direction.AFFERENT

    return result


def _read_nrn(h5f, props, nrn_idx, result):
    # pylint: disable=missing-docstring
    by_gid = group_by_first(nrn_idx)
    for gid, idx in by_gid.items():
        gid_data = h5f[gid2str(gid)][idx]
        for prop, (column_idx, _) in props:
            result.loc[gid, prop] = normalize_endianness(gid_data[:, column_idx])

    for prop, (_, func) in props:
        if func == float:
            # already normalized_endianness; no change needed
            continue

        if func == int:
            def transform(xs):
                return xs.values.astype(int)
        else:
            def transform(xs):
                return list(map(func, xs))  # pylint: disable=cell-var-from-loop
        result[prop] = transform(result[prop])


def _deduce_nrn_prefix(nrn_path):
    if ":" in nrn_path:
        pop, f = nrn_path.split(":", 1)
        raise BluePyError(f"Could not define a population : {pop} for the nrn files : {f}")
    if nrn_path.endswith('nrn.h5'):
        if os.path.exists(nrn_path):
            return nrn_path[:-len('nrn.h5')]
    for prefix in ['', 'proj_']:
        if os.path.exists(os.path.join(nrn_path, prefix + 'nrn.h5')):
            return os.path.join(nrn_path, prefix)
    raise BluePyError(f"Could not find nrn.h5 file at {nrn_path}")


class NrnConnectome:
    """ Connectome access via nrn*.h5 files. """

    def __init__(self, nrn_path, circuit):
        self._prefix = _deduce_nrn_prefix(nrn_path)
        self._circuit = circuit

    @cached_property
    def available_properties(self):
        """ Set of available Synapse properties. """
        result = set(NRN_PROPERTIES).union(set(NRN_POSITIONS))
        if self._version < 5:
            result.remove(Synapse.NRRP)
        return result

    def _open_nrn(self, suffix=''):
        return h5py.File(f"{self._prefix}nrn{suffix}.h5", 'r')

    @cached_property
    def _version(self):
        """ NRN revision number. """
        with self._open_nrn() as h5f:
            try:
                attr = h5f['info'].attrs['version']
                return ensure_list(attr)[0]
            except (KeyError, IndexError):
                return 0

    def _nrn_synapse_properties(self, nrn_idx, properties, direction):
        # pylint: disable=missing-docstring, too-many-branches
        properties = set(properties)
        if Synapse.NRRP in properties:
            required_version = 5
            if self._version < required_version:
                raise BluePyError(
                    f"Trying to fetch Synapse.NRRP from NRN version {self._version} "
                    f"(required: >={required_version})"
                )

        nrn_properties = []
        nrn_positions = []
        for prop in properties:
            if (prop == Synapse.POST_GID) and (direction == Direction.AFFERENT):
                continue
            if (prop == Synapse.PRE_GID) and (direction == Direction.EFFERENT):
                continue
            if prop in NRN_PROPERTIES:
                nrn_properties.append((prop, NRN_PROPERTIES[prop]))
            elif prop in NRN_POSITIONS:
                nrn_positions.append((prop, NRN_POSITIONS[prop]))
            else:
                raise BluePyError(f"Unknown property: {prop}")

        result = pd.DataFrame(index=pd.MultiIndex.from_tuples(nrn_idx))

        if direction == Direction.AFFERENT:
            if nrn_properties:
                with self._open_nrn() as h5f:
                    _read_nrn(h5f, nrn_properties, nrn_idx, result)

            if nrn_positions:
                with self._open_nrn('_positions') as h5f:
                    _read_nrn(h5f, nrn_positions, nrn_idx, result)

            if Synapse.POST_GID in properties:
                result[Synapse.POST_GID] = result.index.get_level_values(0)
        else:
            if nrn_properties:
                with self._open_nrn('_efferent') as h5f:
                    _read_nrn(h5f, nrn_properties, nrn_idx, result)

            if nrn_positions:
                with self._open_nrn('_positions_efferent') as h5f:
                    _read_nrn(h5f, nrn_positions, nrn_idx, result)

            if Synapse.PRE_GID in properties:
                result[Synapse.PRE_GID] = result.index.get_level_values(0)

        return result

    def _synapse_properties(self, nrn_idx, properties, direction):
        if len(nrn_idx) < 1:
            return pd.DataFrame(columns=properties)

        pre_morph_properties = {
            Synapse.PRE_SECTION_DISTANCE, Synapse.PRE_NEURITE_DISTANCE
        }.intersection(properties)
        post_morph_properties = {
            Synapse.POST_SECTION_DISTANCE, Synapse.POST_NEURITE_DISTANCE
        }.intersection(properties)

        nrn_properties = set(properties) - pre_morph_properties - post_morph_properties
        if pre_morph_properties:
            nrn_properties.update([
                Synapse.PRE_GID,
                Synapse.PRE_SECTION_ID, Synapse.PRE_SEGMENT_ID, Synapse.PRE_SEGMENT_OFFSET,
            ])
        if post_morph_properties:
            nrn_properties.update([
                Synapse.POST_GID,
                Synapse.POST_SECTION_ID, Synapse.POST_SEGMENT_ID, Synapse.POST_SEGMENT_OFFSET,
            ])
        if Synapse.TOUCH_DISTANCE in properties:
            nrn_properties.remove(Synapse.TOUCH_DISTANCE)
            nrn_properties.update([
                Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR, Synapse.PRE_Z_CONTOUR,
                Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR, Synapse.POST_Z_CONTOUR,
            ])

        result = self._nrn_synapse_properties(nrn_idx, nrn_properties, direction=direction)

        if pre_morph_properties:
            morph_features = result.rename(columns={
                Synapse.PRE_GID: Cell.ID,
                Synapse.PRE_SECTION_ID: Section.ID,
                Synapse.PRE_SEGMENT_ID: Segment.ID,
            }).groupby(Cell.ID, group_keys=False).apply(_calc_morph_features, self._circuit)
            if Synapse.PRE_SECTION_DISTANCE in pre_morph_properties:
                result[Synapse.PRE_SECTION_DISTANCE] = (
                    morph_features[Segment.SECTION_START_DISTANCE] +
                    result[Synapse.PRE_SEGMENT_OFFSET]
                )
            if Synapse.PRE_NEURITE_DISTANCE in pre_morph_properties:
                result[Synapse.PRE_NEURITE_DISTANCE] = (
                    morph_features[Section.NEURITE_START_DISTANCE] +
                    morph_features[Segment.SECTION_START_DISTANCE] +
                    result[Synapse.PRE_SEGMENT_OFFSET]
                )
        if post_morph_properties:
            morph_features = result.rename(columns={
                Synapse.POST_GID: Cell.ID,
                Synapse.POST_SECTION_ID: Section.ID,
                Synapse.POST_SEGMENT_ID: Segment.ID,
            }).groupby(Cell.ID, group_keys=False).apply(_calc_morph_features, self._circuit)
            if Synapse.POST_SECTION_DISTANCE in post_morph_properties:
                result[Synapse.POST_SECTION_DISTANCE] = (
                    morph_features[Segment.SECTION_START_DISTANCE] +
                    result[Synapse.POST_SEGMENT_OFFSET].clip(lower=0.0)
                )
            if Synapse.POST_NEURITE_DISTANCE in post_morph_properties:
                result[Synapse.POST_NEURITE_DISTANCE] = (
                    morph_features[Section.NEURITE_START_DISTANCE] +
                    morph_features[Segment.SECTION_START_DISTANCE] +
                    result[Synapse.POST_SEGMENT_OFFSET].clip(lower=0.0)
                )

        if Synapse.TOUCH_DISTANCE in properties:
            pre_xyz = [Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR, Synapse.PRE_Z_CONTOUR]
            post_xyz = [Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR, Synapse.POST_Z_CONTOUR]
            result[Synapse.TOUCH_DISTANCE] = np.linalg.norm(
                result[pre_xyz].values - result[post_xyz].values,
                axis=1
            )

        return result[properties]

    def synapse_properties(self, synapse_ids, properties):
        """ Synapse properties as pandas DataFrame. """
        return self._synapse_properties(synapse_ids, properties, direction=Direction.AFFERENT)

    def afferent_gids(self, gid):
        """ Sorted array of unique presynaptic GIDs for given `gid`. """
        with self._open_nrn('_summary') as h5f:
            key = gid2str(gid)
            if key in h5f:
                dset = h5f[key]
                return np.sort(dset[:, 0][dset[:, 2] > 0]).astype(int)
            else:
                return np.empty(0)

    def efferent_gids(self, gid):
        """ Sorted array of unique postsynaptic GIDs for given `gid`. """
        with self._open_nrn('_summary') as h5f:
            key = gid2str(gid)
            if key in h5f:
                dset = h5f[key]
                return np.sort(dset[:, 0][dset[:, 1] > 0]).astype(int)
            else:
                return np.empty(0)

    def pathway_synapses(self, pre_gids, post_gids, properties):
        """ Get synapses corresponding to `pre_gids` -> `post_gids` connections. """
        # pylint: disable=too-many-locals, too-many-branches
        direction = _choose_optimal_layout_direction(pre_gids, post_gids)

        if direction == Direction.AFFERENT:
            suffix, key_gids, connected_gids = '', post_gids, pre_gids
        else:
            suffix, key_gids, connected_gids = '_efferent', pre_gids, post_gids

        nrn_idx = []
        synapse_ids = []
        with self._open_nrn(suffix) as h5f:
            for key_gid in key_gids:
                key = gid2str(key_gid)
                if key in h5f:
                    dset = h5f[key]
                    idx = np.arange(len(dset))
                    if connected_gids is None:
                        mask = None
                    else:
                        mask = np.in1d(np.asarray(dset[:, 0], dtype=int), connected_gids)
                        idx = idx[mask]
                    ids = [(key_gid, k) for k in idx]
                    if direction == Direction.AFFERENT:
                        synapse_ids.extend(ids)
                    else:
                        nrn_idx.extend(ids)
                        dset_idx = h5f[key + "_afferentIndices"]
                        ids = np.stack([
                            np.asarray(dset[:, 0], dtype=int),
                            dset_idx[:, 0]
                        ]).T
                        if mask is not None:
                            ids = ids[mask]
                        synapse_ids.extend(map(tuple, ids))

        if direction == Direction.AFFERENT:
            nrn_idx = synapse_ids

        if properties is None:
            return synapse_ids
        else:
            result = self._synapse_properties(nrn_idx, properties, direction=direction)
            if len(synapse_ids) > 0:
                result.index = pd.MultiIndex.from_tuples(synapse_ids)
                result.sort_index(inplace=True)

        return result

    def iter_connections(self, pre_gids, post_gids, unique_gids, shuffle):
        """
        Iterate through `pre_gids` -> `post_gids` connections.

        If `unique_gids` is set to True, no gid would be used more than once.
        If `shuffle` is set to True, result order would be (somewhat) randomised.

        Yields (pre_gid, post_gid, synapse_count) tuples.
        """
        # pylint: disable=too-many-locals,too-many-branches

        direction = _choose_optimal_layout_direction(pre_gids, post_gids)
        if direction == Direction.AFFERENT:
            gid_keys, include_gids = post_gids, pre_gids
        else:
            gid_keys, include_gids = pre_gids, post_gids

        gid_keys = np.unique(gid_keys)

        if shuffle:
            gid_keys = np.random.permutation(gid_keys)

        # this ensures the same behavior as the sonata impl with unique and sorted gids.
        if include_gids is not None:
            include_gids = np.unique(include_gids)

        exclude_gids = set()

        with self._open_nrn('_summary') as h5f:
            for gid in gid_keys:
                key = gid2str(gid)
                if key not in h5f:
                    continue
                # slicing h5py dataset with a boolean row mask is slow => copy to NumPy array
                dset = np.asarray(h5f[key][:], dtype=int)
                if direction == Direction.AFFERENT:
                    dset = dset[dset[:, 2] > 0]
                else:
                    dset = dset[dset[:, 1] > 0]
                if include_gids is not None:
                    dset = dset[np.in1d(dset[:, 0], include_gids)]
                if shuffle:
                    dset = np.random.permutation(dset)
                for row in dset:
                    gid2 = row[0]
                    if unique_gids and (gid2 in exclude_gids):
                        continue
                    if direction == Direction.AFFERENT:
                        yield gid2, gid, row[2]
                    else:
                        yield gid, gid2, row[1]
                    if unique_gids:
                        exclude_gids.add(gid2)
                        break
