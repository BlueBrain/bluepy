""" Circuit stats helper. """

import itertools
import logging

import numpy as np
import pandas as pd
from morphio import SectionType

from bluepy import morphology
from bluepy.exceptions import BluePyError
from bluepy.utils import ensure_list
from bluepy.enums import Cell, Section, Segment, Synapse


L = logging.getLogger(__name__)


def _segment_lengths(segments):
    """ Find segment lengths given a DataFrame returned by morph.spatial_index() query. """
    return np.linalg.norm(
        segments[[Segment.X1, Segment.Y1, Segment.Z1]].values -
        segments[[Segment.X2, Segment.Y2, Segment.Z2]].values,
        axis=1
    )


def _segment_volumes(segments):
    """ Find segment volumes given a DataFrame returned by morph.spatial_index() query. """
    r1 = segments[Segment.R1]
    r2 = segments[Segment.R2]
    h = _segment_lengths(segments)
    return np.pi * h * ((r1 * r1) + (r1 * r2) + (r2 * r2)) / 3.0


def _count_regions_per_points(points, annotation, weights=None):
    """ Find regions for the given points, count number/weight of points per region. """
    if len(points) == 0:
        return {}
    ids = annotation.lookup(points, outer_value=0)
    if weights is None:
        values, counts = np.unique(ids, return_counts=True)
        result = dict(zip(values, counts))
    else:
        result = {}
        values, index = np.unique(ids, return_inverse=True)
        for k, v in enumerate(values):
            result[v] = np.sum(weights[index == k])
    return result


class StatsHelper:
    """ Circuit stats helper. """
    def __init__(self, circuit):
        self._circuit = circuit

    def cell_density(self, roi, group=None):
        """
        Cell density within given region of interest.

        Args:
            roi: region of interest (bluepy.geometry.roi.ROI instance)
            group: cell group of interest (if None, all cells are taken into the account)

        Returns:
            Cell count within `roi` / volume (unit: 1 / mm^3)
        """
        # TODO: use spatial index if accessible
        p0, p1 = roi.bbox
        xyz = self._circuit.cells.get(
            {
                Cell.X: (p0[0], p1[0]),
                Cell.Y: (p0[1], p1[1]),
                Cell.Z: (p0[2], p1[2]),
            },
            properties=[Cell.X, Cell.Y, Cell.Z]
        )
        if group is not None:
            gids = self._circuit.cells.ids(group)
            xyz = xyz[xyz.index.isin(gids)]
        cell_count = np.count_nonzero(roi.contains(xyz.values))
        return 1e9 * cell_count / roi.volume

    def synapse_density(self, roi):
        """
        Synapse density within given region of interest.

        Args:
            roi: region of interest (bluepy.geometry.roi.ROI instance)

        Returns:
            Synapse count within `roi` / volume (unit: 1 / mm^3)
        """
        p0, p1 = roi.bbox
        synapses = self._circuit.connectome.spatial_index.q_window(p0, p1)
        midpoints = 0.5 * (
            synapses[[Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER, Synapse.PRE_Z_CENTER]].values +
            synapses[[Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]].values
        )
        synapse_count = np.count_nonzero(roi.contains(midpoints))
        return 1e9 * synapse_count / roi.volume

    def fibre_density(self, roi, neurite_types=None):
        """
        Fibre density within given region of interest.

        Args:
            roi: region of interest (bluepy.geometry.roi.ROI instance)
            neurite_types (list/morphio.SectionType): morphio.SectionType or list of
            morphio.SectionType
            (if None, all neurites are taken into account)

        Returns:
            Ratio of `roi` volume occupied by neurites of the specified type(s).
        """
        p0, p1 = roi.bbox
        segments = self._circuit.morph.spatial_index.q_window(p0, p1)
        if neurite_types is not None:
            neurite_types = ensure_list(neurite_types)
            segments = segments[segments[Section.NEURITE_TYPE].isin(neurite_types)]
        mask = np.logical_and(  # pylint: disable=assignment-from-no-return
            roi.contains(segments[[Segment.X1, Segment.Y1, Segment.Z1]].values),
            roi.contains(segments[[Segment.X2, Segment.Y2, Segment.Z2]].values)
        )
        segments = segments[mask]
        segments_volume = np.sum(_segment_volumes(segments))
        return segments_volume / roi.volume

    def sample_divergence(self, pre, post, by, sample=None):
        """
        `pre` -> `post` divergence.

        Args:
            pre: presynaptic cell group
            post: postsynaptic cell group
            by: '(syn)apses' | '(conn)ections'
            sample: sample size for presynaptic group

        Returns:
            Array with synapse / connection count per each cell from `pre` sample
            (taking into account only connections to cells in `post`).
        """
        by_alternatives = {'syn', 'synapses', 'conn', 'connections'}
        if by not in by_alternatives:
            raise BluePyError(f"`by` should be one of {by_alternatives}; got: {by}")

        pre_sample = self._circuit.cells.ids(pre, sample=sample)

        result = {gid: 0 for gid in pre_sample}
        if by in ('syn', 'synapses'):
            connections = self._circuit.connectome.iter_connections(
                pre_sample, post, return_synapse_count=True
            )
            for pre_gid, _, synapse_count in connections:
                result[pre_gid] += synapse_count
        else:
            connections = self._circuit.connectome.iter_connections(pre_sample, post)
            for pre_gid, _ in connections:
                result[pre_gid] += 1

        return np.array(list(result.values()))

    def sample_convergence(self, pre, post, by=None, sample=None):
        """
        `pre` -> `post` convergence.

        Args:
            pre: presynaptic cell group
            post: postsynaptic cell group
            by: '(syn)apses' | '(conn)ections'
            sample: sample size for postsynaptic group

        Returns:
            Array with synapse / connection count per each cell from `post` sample
            (taking into account only connections from cells in `pre`).
        """
        by_alternatives = {'syn', 'synapses', 'conn', 'connections'}
        if by not in by_alternatives:
            raise BluePyError(f"`by` should be one of {by_alternatives}; got: {by}")

        post_sample = self._circuit.cells.ids(post, sample=sample)

        result = {gid: 0 for gid in post_sample}
        if by in ('syn', 'synapses'):
            connections = self._circuit.connectome.iter_connections(
                pre, post_sample, return_synapse_count=True
            )
            for _, post_gid, synapse_count in connections:
                result[post_gid] += synapse_count
        else:
            connections = self._circuit.connectome.iter_connections(pre, post_sample)
            for _, post_gid in connections:
                result[post_gid] += 1

        return np.array(list(result.values()))

    def segment_region_distribution(self, annotation, cell_group, normalize=False, by='count'):
        """ Segment distribution by regions. """
        # pylint: disable=too-many-locals
        by_alternatives = ('count', 'length', 'volume')
        if by not in by_alternatives:
            raise ValueError(
                f"Invalid 'by' argument: '{by}', should be one of {by_alternatives}"
            )

        def _iter_on(sections, n_type=None):
            yield from filter(lambda x: (n_type is None or x.type == n_type), sections)

        index = []
        result = []
        for gid in self._circuit.cells.ids(group=cell_group):
            morph = self._circuit.morph.get(gid, transform=True)
            neurite_types = [SectionType.apical_dendrite,
                             SectionType.axon,
                             SectionType.basal_dendrite]
            for neurite_type in neurite_types:
                points = np.array(
                    [p for section in _iter_on(morph.sections, n_type=neurite_type)
                     for p in morphology.segment_midpoints(section)])
                if by == 'length':
                    weights = np.array(
                        [length for section in _iter_on(morph.sections, n_type=neurite_type)
                         for length in morphology.segment_lengths(section)])
                elif by == 'volume':
                    weights = np.array(
                        [volume for section in _iter_on(morph.sections, n_type=neurite_type)
                         for volume in morphology.segment_volumes(section)])
                else:
                    weights = None
                values = _count_regions_per_points(points, annotation, weights)
                index.append((gid, neurite_type.name))
                result.append(values)

        index = pd.MultiIndex.from_tuples(index, names=['gid', 'branch_type'])
        result = pd.DataFrame(result, index=index).fillna(0).sort_index()

        if normalize:
            result = result.div(result.sum(axis=1), axis=0).astype(float)
        elif by == 'count':
            result = result.astype(int)
        else:
            result = result.astype(float)
        return result

    def synapse_region_distribution(
        self, annotation, side, groupby, pre=None, post=None, normalize=False
    ):
        """
        Synapse distribution by regions.

        Args:
            annotation: VoxelData with region annotation
            side: 'pre' | 'post' (-synaptic)
            groupby: cell property to group by
            pre: presynaptic cell group
            post: postsynaptic cell group
            normalize: if True, return fractions instead of synapse number

        Returns:
            Pandas DataFrame indexed by `groupby` with regions as columns.
        """
        if side == 'pre':
            gid_column = Synapse.PRE_GID
            xyz_columns = [Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER, Synapse.PRE_Z_CENTER]
        elif side == 'post':
            gid_column = Synapse.POST_GID
            xyz_columns = [Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]
        else:
            raise BluePyError(f"`side` should be one of {{'pre', 'post'}}; got: {side}")

        df = self._circuit.connectome.pathway_synapses(
            pre=pre, post=post, properties=[gid_column] + xyz_columns
        )

        groupby_key = df[gid_column]
        if groupby != Cell.ID:
            prop = self._circuit.cells.get(properties=groupby)
            groupby_key = groupby_key.map(prop)

        df2 = pd.DataFrame({
            'region': annotation.lookup(df[xyz_columns].values, outer_value=-1),
            groupby: groupby_key.values,
        })

        result = df2.groupby([groupby, 'region']).size().unstack(fill_value=0)
        if normalize:
            result = result.div(result.sum(axis=1), axis=0).astype(float)

        return result

    def _get_region_mask(self, region):
        if region is None:
            return None
        else:
            return self._circuit.atlas.get_region_mask(region)

    def _calc_bouton_density(self, gid, synapses_per_bouton, region_mask):
        """ Calculate bouton density for a given `gid`. """
        if region_mask is None:
            # count all efferent synapses and total axon length
            synapse_count = len(self._circuit.connectome.efferent_synapses(gid))
            morph = self._circuit.morph.get(gid, transform=False)
            axon_length = sum(morphology.section_length(s) for s in
                              filter(lambda x: x.type == SectionType.axon, morph.sections))
        else:
            # find all segments which endpoints fall into the target region
            all_pts = self._circuit.morph.segment_points(
                gid, transform=True, neurite_type=SectionType.axon
            )
            mask1 = region_mask.lookup(
                all_pts[[Segment.X1, Segment.Y1, Segment.Z1]].values, outer_value=False
            )
            mask2 = region_mask.lookup(
                all_pts[[Segment.X2, Segment.Y2, Segment.Z2]].values, outer_value=False
            )
            filtered = all_pts[mask1 & mask2]
            if filtered.empty:
                L.warning("No axon segments found inside target region for GID %d", gid)
                return np.nan

            # total length for those filtered segments
            axon_length = _segment_lengths(filtered).sum()

            # find axon segments with synapses; count synapses per each such segment
            INDEX_COLS = [Synapse.PRE_SECTION_ID, '_PRE_SEGMENT_ID']
            syn_per_segment = self._circuit.connectome.efferent_synapses(
                gid, properties=INDEX_COLS
            ).groupby(INDEX_COLS).size()

            # count synapses on filtered segments
            labels = filtered.index
            synapse_count = syn_per_segment[syn_per_segment.index.intersection(labels)].sum()

        return (1.0 * synapse_count / synapses_per_bouton) / axon_length

    def bouton_density(self, gid, synapses_per_bouton=1.0, region=None):
        """ Calculate bouton density for a given `gid`. """
        region_mask = self._get_region_mask(region)
        return self._calc_bouton_density(gid, synapses_per_bouton, region_mask)

    def sample_bouton_density(self, n, group=None, synapses_per_bouton=1.0, region=None):
        """
        Sample bouton density.

        Args:
            n: sample size
            group: cell group
            synapses_per_bouton: assumed number of synapses per bouton

        Returns:
            numpy array of length min(n, N) with bouton density per cell,
            where N is the total number cells in the specified cell group.
        """
        gids = self._circuit.cells.ids(group)
        if len(gids) > n:
            gids = np.random.choice(gids, size=n, replace=False)
        region_mask = self._get_region_mask(region)
        return np.array([
            self._calc_bouton_density(gid, synapses_per_bouton, region_mask) for gid in gids
        ])

    def sample_pathway_synapse_count(self, n, pre=None, post=None, unique_gids=False):
        """
        Sample synapse count for pathway connections.

        Args:
            n: sample size
            pre: presynaptic cell group
            post: postsynaptic cell group
            unique_gids(bool): don't use one GID more than once

        Returns:
            numpy array of length min(n, N) with synapse number per connection,
            where N is the total number of connections satisfying the constraints.
        """
        it = self._circuit.connectome.iter_connections(
            pre, post, shuffle=True, unique_gids=unique_gids, return_synapse_count=True
        )
        return np.array([p[2] for p in itertools.islice(it, n)])
