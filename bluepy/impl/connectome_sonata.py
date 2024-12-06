"""
Connectome access via SONATA Edges files.

https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#neuron_networks_edges
"""
import os
import numpy as np
import pandas as pd
import h5py
from morphio import SectionType

import libsonata
from cached_property import cached_property

from bluepy.exceptions import BluePyError
from bluepy.enums import Cell, Section, Segment, Synapse
from bluepy.impl.utils import (
    DYNAMICS_PREFIX,
    _inc,
    _dec,
    _calc_morph_features,
    add_dynamic_prefix,
    ensure_ids,
)
from bluepy.utils import ensure_list

NEURITE_TYPE_MAP = {
    0: SectionType.soma,
    1: SectionType.axon,
    2: SectionType.basal_dendrite,
    3: SectionType.apical_dendrite,
}

SYNAPSE_PROPERTIES = {

    Synapse.PRE_GID: "@source_node",
    Synapse.POST_GID: "@target_node",

    Synapse.AXONAL_DELAY: 'delay',
    Synapse.D_SYN: 'depression_time',
    Synapse.F_SYN: 'facilitation_time',
    Synapse.DTC: 'decay_time',
    Synapse.G_SYNX: 'conductance',
    Synapse.U_SYN: 'u_syn',
    Synapse.TYPE: 'syn_type_id',
    Synapse.NRRP: 'n_rrp_vesicles',

    # presynaptic touch position (in the center of the segment)
    Synapse.PRE_X_CENTER: "efferent_center_x",
    Synapse.PRE_Y_CENTER: "efferent_center_y",
    Synapse.PRE_Z_CENTER: "efferent_center_z",

    # presynaptic touch position (on the segment surface)
    Synapse.PRE_X_CONTOUR: "efferent_surface_x",
    Synapse.PRE_Y_CONTOUR: "efferent_surface_y",
    Synapse.PRE_Z_CONTOUR: "efferent_surface_z",

    # postsynaptic touch position (in the center of the segment)
    Synapse.POST_X_CENTER: "afferent_center_x",
    Synapse.POST_Y_CENTER: "afferent_center_y",
    Synapse.POST_Z_CENTER: "afferent_center_z",

    # postsynaptic touch position (on the segment surface)
    Synapse.POST_X_CONTOUR: "afferent_surface_x",
    Synapse.POST_Y_CONTOUR: "afferent_surface_y",
    Synapse.POST_Z_CONTOUR: "afferent_surface_z",

    # specific to sonata circuits
    # pylint: disable=no-member
    Synapse.CONDUCTANCE_RATIO: Synapse.CONDUCTANCE_RATIO.value,
    Synapse.U_HILL_COEFFICIENT: Synapse.U_HILL_COEFFICIENT.value
}

OLD_MORPH_SYNAPSE_PROPERTIES = {
    Synapse.POST_BRANCH_ORDER: 'morpho_branch_order_post',
    Synapse.POST_BRANCH_TYPE: 'morpho_section_type_post',
    Synapse.POST_SECTION_ID: 'morpho_section_id_post',
    Synapse.POST_SEGMENT_ID: 'morpho_segment_id_post',
    Synapse.POST_SEGMENT_OFFSET: 'morpho_offset_segment_post',

    Synapse.PRE_BRANCH_ORDER: 'morpho_branch_order_pre',
    Synapse.PRE_SECTION_ID: 'morpho_section_id_pre',
    Synapse.PRE_SEGMENT_ID: 'morpho_segment_id_pre',
    Synapse.PRE_SEGMENT_OFFSET: 'morpho_offset_segment_pre',
}

MORPH_SYNAPSE_PROPERTIES = {
    Synapse.POST_BRANCH_TYPE: 'afferent_section_type',
    Synapse.POST_SECTION_ID: 'afferent_section_id',
    Synapse.POST_SEGMENT_ID: 'afferent_segment_id',
    Synapse.POST_SEGMENT_OFFSET: 'afferent_segment_offset',

    Synapse.PRE_SECTION_ID: 'efferent_section_id',
    Synapse.PRE_SEGMENT_ID: 'efferent_segment_id',
    Synapse.PRE_SEGMENT_OFFSET: 'efferent_segment_offset',
}

PRE_MORPH_MANDATORY = {Synapse.PRE_SECTION_ID, Synapse.PRE_SEGMENT_ID, Synapse.PRE_SEGMENT_OFFSET}
POST_MORPH_MANDATORY = {Synapse.POST_SECTION_ID, Synapse.POST_SEGMENT_ID,
                        Synapse.POST_SEGMENT_OFFSET}
TOUCH_DISTANCE_MANDATORY = {Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR, Synapse.PRE_Z_CONTOUR,
                            Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR, Synapse.POST_Z_CONTOUR}


def _estimate_range_size(func, node_ids, n=3):
    """Median size of index second level for some node IDs from the provided list.

    Node IDs are assumed to be in SONATA (0-based) format."""
    assert len(node_ids) > 0
    if len(node_ids) > n:
        node_ids = np.random.choice(node_ids, size=n, replace=False)
    return np.median([len(func(node_id).ranges) for node_id in node_ids])


def _check_sonata_file(filepath):
    """Check if a h5 file is a sonata file.

    Notes:
        This is done to prevent some libhdf5 error messages when failing to open a hdf5 file.
    """

    def _check_node_population(h5f, pop):
        for dataset in ["source_node_id", "target_node_id"]:
            try:
                h5f[f"edges/{pop}/{dataset}"].attrs["node_population"]
            except KeyError:
                raise BluePyError(
                    "Missing the 'node_population' attribute in the "
                    f"{filepath}/edges/{pop}/{dataset} dataset. See: "
                    "https://sonata-extension.readthedocs.io/en/latest"
                    "/faq.html#what-do-i-do-about-a-missing-the-node-population-attribute-errors")

    with h5py.File(filepath, "r") as h5:
        if "edges" not in h5:
            raise BluePyError(f"{filepath} is not a sonata edge file")
        for population in h5["edges"]:
            _check_node_population(h5, population)
    return filepath


def _resolve_sonata_path(path):
    """Find the sonata edge file."""
    if os.path.isfile(path):
        return path
    # from neurodamus specs the only possible file names, if one uses a directory as path, are
    # "edges.sonata", "edges.h5"
    filenames = ["edges.sonata", "edges.h5"]
    for filename in filenames:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            return filepath
    raise BluePyError(f"Can't find a sonata file path from {path}")


def _ensure_none_or_list(v):
    return None if v is None else ensure_list(v)


def _is_empty(xs):
    return (xs is not None) and (len(xs) == 0)


def _optimal_direction(population, source_node_ids, target_node_ids):
    """Choose between source and target node IDs for iterating."""
    if source_node_ids is None:
        return "target"
    if target_node_ids is None:
        return "source"

    # Checking the indexing 'direction'. One direction has contiguous indices.
    range_size_source = _estimate_range_size(population.efferent_edges, source_node_ids)
    range_size_target = _estimate_range_size(population.afferent_edges, target_node_ids)

    return "source" if (range_size_source < range_size_target) else "target"


class SonataConnectome:
    """ Connectome access via SONATA Edges files. """

    def __init__(self, filepath, circuit):
        try:
            filepath, population = filepath.split(":")
        except ValueError:
            population = None

        if population is None:
            edge_storage = libsonata.EdgeStorage(_check_sonata_file(filepath))
            populations = list(edge_storage.population_names)
            if len(populations) != 1:
                raise BluePyError(
                    "Only single-population SONATA file are supported if no population "
                    "is provided in the BlueConfig file via: nrnPath filepath:population_name.")
            population = populations[0]

        self._path = filepath
        self._population_name = population
        self._circuit = circuit
        self._property_dict = self._set_property_dict()

    @property
    def _population(self):
        edge_storage = libsonata.EdgeStorage(_check_sonata_file(self._path))
        return edge_storage.open_population(self._population_name)

    @cached_property
    def _attribute_names(self):
        return set(self._population.attribute_names)

    @cached_property
    def _dynamics_params_names(self):
        return set(add_dynamic_prefix(self._population.dynamics_attribute_names))

    @property
    def _topology_property_names(self):
        return {SYNAPSE_PROPERTIES[Synapse.PRE_GID], SYNAPSE_PROPERTIES[Synapse.POST_GID]}

    @property
    def property_names(self):
        """Set of available edge properties.

        Notes:
            Properties are a combination of the group attributes, the dynamics_params and the
            topology properties.
        """
        return self._attribute_names | self._dynamics_params_names | self._topology_property_names

    def _get_property(self, prop, selection):
        if prop == SYNAPSE_PROPERTIES[Synapse.PRE_GID]:
            result = ensure_ids(self._population.source_nodes(selection))
        elif prop == SYNAPSE_PROPERTIES[Synapse.POST_GID]:
            result = ensure_ids(self._population.target_nodes(selection))
        elif prop in self._attribute_names:
            result = self._population.get_attribute(prop, selection)
        elif prop in self._dynamics_params_names:
            result = self._population.get_dynamics_attribute(
                prop.split(DYNAMICS_PREFIX)[1], selection
            )
        else:
            raise BluePyError(f"No such property: {prop}")
        return result

    def _pathway_edges(self, source_node_ids=None, target_node_ids=None):
        """Get edges corresponding to ``source`` -> ``target`` connections.

        Args:
            source (int/sequence/None): source node ids
            target (int/sequence/None): target node ids

        Returns:
            List of edge IDs
        """
        source_node_ids = _ensure_none_or_list(source_node_ids)
        target_node_ids = _ensure_none_or_list(target_node_ids)

        if source_node_ids is None:
            selection = self._population.afferent_edges(target_node_ids)
        elif target_node_ids is None:
            selection = self._population.efferent_edges(source_node_ids)
        else:
            selection = self._population.connecting_edges(source_node_ids, target_node_ids)

        return ensure_ids(selection.flatten())

    def _iter_connections(self, source_node_ids, target_node_ids, unique_node_ids, shuffle):
        """Iterate through `source_node_ids` -> `target_node_ids` connections.

        Args:
            source_node_ids (sequence/None): source node ids
            target_node_ids (sequence/None): target node ids
            unique_node_ids: if True, no node ID will be used more than once as source or
                target for edges. Careful, this flag does not provide unique (source, target)
                pairs but unique node IDs.
            shuffle: if True, result order would be (somewhat) randomized

        Yields:
            (source_node_id, target_node_id, edge_ids)
        """
        # pylint: disable=too-many-branches,too-many-locals
        if _is_empty(source_node_ids) or _is_empty(target_node_ids):
            return

        direction = _optimal_direction(self._population, source_node_ids, target_node_ids)
        if direction == "target":
            primary_node_ids, secondary_node_ids = target_node_ids, source_node_ids
            get_connected_node_ids = self._afferent_nodes
        else:
            primary_node_ids, secondary_node_ids = source_node_ids, target_node_ids
            get_connected_node_ids = self._efferent_nodes

        primary_node_ids = np.unique(primary_node_ids)
        if shuffle:
            np.random.shuffle(primary_node_ids)

        if secondary_node_ids is not None:
            secondary_node_ids = np.unique(secondary_node_ids)

        secondary_node_ids_used = set()

        for key_node_id in primary_node_ids:
            connected_node_ids = get_connected_node_ids(key_node_id, unique=False)
            # [[secondary_node_id, count], ...]
            connected_node_ids_with_count = np.stack(
                np.unique(connected_node_ids, return_counts=True)
            ).transpose()
            # np.stack(uint64, int64) -> float64
            connected_node_ids_with_count = connected_node_ids_with_count.astype(np.uint32)
            if secondary_node_ids is not None:
                mask = np.in1d(
                    connected_node_ids_with_count[:, 0], secondary_node_ids, assume_unique=True
                )
                connected_node_ids_with_count = connected_node_ids_with_count[mask]
            if shuffle:
                np.random.shuffle(connected_node_ids_with_count)

            for conn_node_id, edge_count in connected_node_ids_with_count:
                if unique_node_ids and (conn_node_id in secondary_node_ids_used):
                    continue
                if direction == "target":
                    yield conn_node_id, key_node_id, edge_count
                else:
                    yield key_node_id, conn_node_id, edge_count
                if unique_node_ids:
                    secondary_node_ids_used.add(conn_node_id)
                    break

    def _afferent_nodes(self, target, unique=True):
        """Get afferent node IDs for given target ``node_id``.

        Notes:
            Afferent nodes are nodes projecting an outgoing edge to one of the ``target`` node.

        Args:
            target (int/sequence): the target node ids you want to use.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Afferent node IDs for all the targets.
        """
        selection = self._population.afferent_edges(target)
        result = self._population.source_nodes(selection)
        if unique:
            result = np.unique(result)
        return ensure_ids(result)

    def _efferent_nodes(self, source, unique=True):
        """Get efferent node IDs for given source ``node_id``.

        Notes:
            Efferent nodes are nodes receiving an incoming edge from one of the ``source`` node.

        Args:
            source (int/sequence): the source ids you want to use.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Efferent node IDs for all the sources.
        """
        selection = self._population.efferent_edges(source)
        result = self._population.target_nodes(selection)
        if unique:
            result = np.unique(result)
        return ensure_ids(result)

    def _set_property_dict(self):
        """ Set the correct Synapse properties mapping for old/new sonata format """
        attr_names = self.property_names
        old_morphology_names = set(OLD_MORPH_SYNAPSE_PROPERTIES.values())
        if any(prop in old_morphology_names for prop in attr_names):
            return {**SYNAPSE_PROPERTIES, **OLD_MORPH_SYNAPSE_PROPERTIES}
        return {**SYNAPSE_PROPERTIES, **MORPH_SYNAPSE_PROPERTIES}

    @cached_property
    def _sonata_available_properties(self):
        inv_map = {v: k for k, v in self._property_dict.items()}
        attributes = {inv_map.get(attr, attr) for attr in self.property_names}
        return attributes

    @cached_property
    def _sonata_topology_property_names(self):
        properties = set()

        if not PRE_MORPH_MANDATORY - self._sonata_available_properties:
            properties.update([Synapse.PRE_SECTION_DISTANCE, Synapse.PRE_NEURITE_DISTANCE])

        if not POST_MORPH_MANDATORY - self._sonata_available_properties:
            properties.update([Synapse.POST_SECTION_DISTANCE, Synapse.POST_NEURITE_DISTANCE])

        if not TOUCH_DISTANCE_MANDATORY - self._sonata_available_properties:
            properties.add(Synapse.TOUCH_DISTANCE)

        return properties

    @cached_property
    def available_properties(self):
        """Set of available properties.

        Notes:
            Historically the enums are used in the dataframe as columns. So in the available
            properties, we return the enums and add the unregistered fields also as simple strings.
        """
        return self._sonata_available_properties | self._sonata_topology_property_names

    def _get_sonata_properties(self, synapse_ids, properties):
        if not isinstance(synapse_ids, np.ndarray):
            synapse_ids = ensure_list(synapse_ids)

        synapse_ids = ensure_ids(synapse_ids)

        selection = libsonata.Selection(synapse_ids)

        res = pd.DataFrame(index=synapse_ids)
        for p in properties:
            prop = self._property_dict.get(p, p)
            res[prop] = self._get_property(prop, selection)

        res.columns = properties

        if Synapse.PRE_GID in res:
            res[Synapse.PRE_GID] += 1
        if Synapse.POST_GID in res:
            res[Synapse.POST_GID] += 1
        if Synapse.POST_BRANCH_TYPE in res:
            res[Synapse.POST_BRANCH_TYPE] = res[Synapse.POST_BRANCH_TYPE].map(NEURITE_TYPE_MAP)
        return res

    def synapse_properties(self, synapse_ids, properties):
        """ Get an array of synapse IDs or DataFrame with synapse properties. """

        # pylint: disable=too-many-branches

        if properties is None:
            return synapse_ids

        if len(synapse_ids) < 1:
            return pd.DataFrame(columns=properties)

        pre_morph_properties = {
            Synapse.PRE_SECTION_DISTANCE, Synapse.PRE_NEURITE_DISTANCE
        }.intersection(properties)
        post_morph_properties = {
            Synapse.POST_SECTION_DISTANCE, Synapse.POST_NEURITE_DISTANCE
        }.intersection(properties)

        sonata_properties = set(properties) - pre_morph_properties - post_morph_properties
        if pre_morph_properties:
            sonata_properties.update([
                Synapse.PRE_GID,
                Synapse.PRE_SECTION_ID, Synapse.PRE_SEGMENT_ID, Synapse.PRE_SEGMENT_OFFSET,
            ])
        if post_morph_properties:
            sonata_properties.update([
                Synapse.POST_GID,
                Synapse.POST_SECTION_ID, Synapse.POST_SEGMENT_ID, Synapse.POST_SEGMENT_OFFSET,
            ])
        if Synapse.TOUCH_DISTANCE in properties:
            if (Synapse.POST_X_CONTOUR not in self.available_properties or
                    Synapse.POST_Y_CONTOUR not in self.available_properties or
                    Synapse.POST_Z_CONTOUR not in self.available_properties):
                raise BluePyError("Cannot compute the touch distance. You need to add the "
                                  "'afferent_surface_[x|y|z]' fields to your sonata files")
            sonata_properties.remove(Synapse.TOUCH_DISTANCE)
            sonata_properties.update([
                Synapse.PRE_X_CONTOUR, Synapse.PRE_Y_CONTOUR, Synapse.PRE_Z_CONTOUR,
                Synapse.POST_X_CONTOUR, Synapse.POST_Y_CONTOUR, Synapse.POST_Z_CONTOUR,
            ])

        try:
            result = self._get_sonata_properties(synapse_ids, sonata_properties)
        except BluePyError as e:
            raise BluePyError(e) from e

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

    def afferent_gids(self, gid, unique=True):
        """ Sorted array of unique presynaptic GIDs for given `gid`. """
        if gid < 1:
            # following the nrn behavior
            return np.empty(0)
        return _inc(self._afferent_nodes(_dec(gid), unique=unique))

    def efferent_gids(self, gid, unique=True):
        """ Sorted array of unique postsynaptic GIDs for given `gid`. """
        if gid < 1:
            # following the nrn behavior
            return np.empty(0)
        return _inc(self._efferent_nodes(_dec(gid), unique=unique))

    def pathway_synapses(self, pre_gids, post_gids, properties):
        """ Synapses corresponding to `pre_gids` -> `post_gids` connections. """
        if pre_gids is None and post_gids is None:
            raise BluePyError("Either `pre_gids` or `post_gids` should be specified")

        ids = self._pathway_edges(_dec(pre_gids), _dec(post_gids))
        return self.synapse_properties(ids, properties)

    @staticmethod
    def _inc_iter(its):
        return ((_inc(source), _inc(target), count) for source, target, count in its)

    def iter_connections(self, pre_gids, post_gids, unique_gids, shuffle):
        """ Iterate through `pre_gids` -> `post_gids` connections. """
        if pre_gids is None and post_gids is None:
            raise BluePyError("Either `pre_gids` or `post_gids` should be specified")

        yield from self._inc_iter(self._iter_connections(_dec(pre_gids), _dec(post_gids),
                                                         unique_node_ids=unique_gids,
                                                         shuffle=shuffle))
