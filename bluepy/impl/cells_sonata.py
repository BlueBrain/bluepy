"""  Access to cell properties for sonata nodes """
import copy
from collections.abc import MutableMapping, Mapping, Sequence
from cached_property import cached_property

import libsonata
import numpy as np
import pandas as pd

from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.impl.utils import (
    DYNAMICS_PREFIX,
    _inc,
    add_dynamic_prefix,
    ensure_ids,
)
from bluepy.utils import ensure_list
from bluepy.utils import query

QUATERNION_ORIENTATION_MANDATORY = {"orientation_w", "orientation_x",
                                    "orientation_y", "orientation_z"}
EULER_ORIENTATION_OPTIONAL = {"rotation_angle_xaxis", "rotation_angle_yaxis",
                              "rotation_angle_zaxis"}


def _check_orientation(property_names):
    quaternion_props = QUATERNION_ORIENTATION_MANDATORY.intersection(property_names)
    euler_props = EULER_ORIENTATION_OPTIONAL.intersection(property_names)

    if (quaternion_props and euler_props) or (len(quaternion_props) in [1, 2, 3]):
        raise BluePyError("Incorrect orientation fields. Should be "
                          "4 quaternions or euler angles or nothing")


def _load_node_sets(targets):
    """Load node sets and filter them with respect to the population."""
    if targets is None:
        return targets
    return targets.to_node_sets(one_based=True)


def _euler2mat(az, ay, ax):
    """Build 3x3 rotation matrices from az, ay, ax rotation angles (in that order).

    Args:
        az: rotation angles around Z (Nx1 NumPy array; radians)
        ay: rotation angles around Y (Nx1 NumPy array; radians)
        ax: rotation angles around X (Nx1 NumPy array; radians)

    Returns:
        List with Nx3x3 rotation matrices corresponding to each of N angle triplets.

    See Also:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix (R = X1 * Y2 * Z3)
    """
    if len(az) != len(ay) or len(az) != len(ax):
        raise BluePyError("All angles must have the same length.")
    c1, s1 = np.cos(ax), np.sin(ax)
    c2, s2 = np.cos(ay), np.sin(ay)
    c3, s3 = np.cos(az), np.sin(az)

    mm = np.array(
        [
            [c2 * c3, -c2 * s3, s2],
            [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
        ]
    )

    return [mm[..., i] for i in range(len(az))]


def _quaternion2mat(aqw, aqx, aqy, aqz):
    """Build 3x3 rotation matrices from quaternions.

    Args:
        aqw: w component of quaternions (Nx1 NumPy array; float)
        aqx: x component of quaternions (Nx1 NumPy array; float)
        aqy: y component of quaternions (Nx1 NumPy array; float)
        aqz: z component of quaternions (Nx1 NumPy array; float)

    Returns:
        List with Nx3x3 rotation matrices corresponding to each of N quaternions.

    See Also:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    def normalize_quaternions(qs):
        """Normalize a bunch of quaternions along axis==1.

        Args:
            qs: quaternions (Nx4 NumPy array; float)

        Returns:
           numpy array of normalized quaternions
        """
        return qs / np.sqrt(np.einsum("...i,...i", qs, qs)).reshape(-1, 1)

    aq = np.dstack([np.asarray(aqw), np.asarray(aqx), np.asarray(aqy), np.asarray(aqz)])[0]
    aq = normalize_quaternions(aq)

    w = aq[:, 0]
    x = aq[:, 1]
    y = aq[:, 2]
    z = aq[:, 3]

    mm = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * x * y - 2 * w * z, 2 * w * y + 2 * x * z],
            [2 * w * z + 2 * x * y, w * w - x * x + y * y - z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * w * x + 2 * y * z, w * w - x * x - y * y + z * z],
        ]
    )

    return [mm[..., i] for i in range(len(aq))]


class SonataCellCollection:
    """Access to cell properties stored in Sonata files."""

    def __init__(self, sonata_path, targets=None):
        self._path = sonata_path
        self._node_sets = _load_node_sets(targets)

        node_storage = libsonata.NodeStorage(sonata_path)
        populations = list(node_storage.population_names)
        if len(populations) != 1:
            raise BluePyError("Only single-population SONATA are supported at the moment")

        self._population_name = populations[0]

        _check_orientation(self._property_names)

    def _check_property(self, prop):
        if prop not in self._data:
            raise BluePyError(f"No such property: '{prop}'")

    @cached_property
    def size(self):
        """Node population size."""
        return self._population.size

    @property
    def _population(self):
        node_storage = libsonata.NodeStorage(self._path)
        return node_storage.open_population(self._population_name)

    @property
    def _property_names(self):
        """Set of available node properties.

        Notes:
            Properties are a combination of the group attributes and the dynamics_params.
        """
        property_names = set(self._population.attribute_names)
        dynamics_params_names = set(add_dynamic_prefix(self._population.dynamics_attribute_names))

        return property_names | dynamics_params_names

    @cached_property
    def _data(self):
        """Collect data for the node population as a pandas.DataFrame."""
        nodes = self._population
        categoricals = nodes.enumeration_names

        _all = nodes.select_all()
        res = pd.DataFrame(index=np.arange(_all.flat_size))

        for attr in sorted(nodes.attribute_names):
            if attr in categoricals:
                enumeration = np.asarray(nodes.get_enumeration(attr, _all))
                values = np.asarray(nodes.enumeration_values(attr))
                # if the size of `values` is large enough compared to `enumeration`, not using
                # categorical reduces the memory usage.
                if values.shape[0] < 0.5 * enumeration.shape[0]:
                    res[attr] = pd.Categorical.from_codes(enumeration, categories=values)
                else:
                    res[attr] = values[enumeration]
            else:
                res[attr] = nodes.get_attribute(attr, _all)
        for attr in sorted(add_dynamic_prefix(nodes.dynamics_attribute_names)):
            res[attr] = nodes.get_dynamics_attribute(attr.split(DYNAMICS_PREFIX)[1], _all)

        def _get(prop):
            return res.get(prop, np.zeros((res.shape[0],)))

        # I cannot use something much better atm like calling self.orientations to fill the
        # dataframe because _data is used in orientation which causes infinite recursion. So this
        # as to be done manually. Also I am not ok to add orientation to snap bc it is bbp specific.
        if not QUATERNION_ORIENTATION_MANDATORY - set(res):
            res[Cell.ORIENTATION] = _quaternion2mat(_get("orientation_w"), _get("orientation_x"),
                                                    _get("orientation_y"), _get("orientation_z"))
        else:
            # orientation always exists if there is no angle and no quat --> unitary matrices
            res[Cell.ORIENTATION] = _euler2mat(_get("rotation_angle_zaxis"),
                                               _get("rotation_angle_yaxis"),
                                               _get("rotation_angle_xaxis"))
        res.index = _inc(res.index)
        return res

    def get_node_set(self, node_set_name):
        """Returns the node set named 'node_set_name'."""
        return self._node_sets[node_set_name]

    def _resolve_ids_from_query(self, group=None, limit=None, sample=None):
        """Node IDs corresponding to node ``group``.

        Args:
            group (int/sequence/str/mapping/None): Which IDs will be
                returned depends on the type of the ``group`` argument:

                - ``int``: return a single node ID if it belongs to the circuit.
                - ``sequence``: return IDs of nodes in an array.
                - ``str``: return IDs of nodes in a node set.
                - ``mapping``: return IDs of nodes matching a properties filter.
                - ``None``: return all node IDs.

                If ``group`` is a ``sequence``, the order of results is preserved.
                Otherwise the result is sorted and contains no duplicates.

            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of the population
                all node IDs are returned.

            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of the NodePopulation then all ids are taken and shuffled.

        Returns:
            numpy.array: A numpy array of IDs.

        Examples:
            The available group parameter values:

            >>> nodes.ids(group=None)  #  returns all IDs
            >>> nodes.ids(group={})  #  returns all IDs
            >>> nodes.ids(group=1)  #  returns the single ID if present in population
            >>> nodes.ids(group=[1,2,3])  # returns list of IDs if all present in population
            >>> nodes.ids(group="node_set_name")  # returns list of IDs matching node set
            >>> nodes.ids(group={ Node.LAYER: 2})  # returns list of IDs matching layer==2
            >>> nodes.ids(group={ Node.LAYER: [2, 3]})  # returns list of IDs with layer in [2,3]
            >>> nodes.ids(group={ Node.X: (0, 1)})  # returns list of IDs with 0 < x < 1
        """
        # pylint: disable=too-many-branches
        preserve_order = False
        if isinstance(group, str):
            group = self.get_node_set(group)

        if group is None:
            result = self._data.index.values
        elif isinstance(group, Mapping):
            result = self._node_ids_by_filter(queries=group)
        elif isinstance(group, np.ndarray):
            result = group
            preserve_order = True
        else:
            result = ensure_list(group)
            preserve_order = isinstance(group, Sequence)

        if sample is not None:
            if len(result) > 0:
                result = np.random.choice(result, min(sample, len(result)), replace=False)
            preserve_order = False
        if limit is not None:
            result = result[:limit]

        result = ensure_ids(result)

        return result if preserve_order else np.unique(result)

    def _node_ids_by_filter(self, queries):
        """Return node IDs if their properties match the `queries` dict.

        `props` values could be:
            pairs (range match for floating dtype fields)
            scalar or iterables (exact or "one of" match for other fields)

        Examples:
            >>> _node_ids_by_filter({ Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' })
            >>> _node_ids_by_filter({ Node.LAYER: [2, 3] })
        """
        # Override filtering for the projections.
        if "node_id" in queries and np.min(queries["node_id"]) > self.size:
            return queries["node_id"]

        return query.gids_by_filter(self._data, queries)

    @property
    def node_sets(self):
        """Return the node sets linked to the population."""
        return set(self._node_sets.keys())

    @property
    def _extra_properties(self):
        """This function is used to add the extra properties that need special checks."""
        return {Cell.ORIENTATION, }

    @cached_property
    def available_properties(self):
        """Set of available node properties."""
        return self._property_names | self._extra_properties

    def ids(self, group=None, limit=None, sample=None):
        """GIDs corresponding to cell `group` (1-based gids).

        Args:
            group (int/sequence/str/mapping/None): Which gids will be returned depends on the type
            of the ``group`` argument:

                - ``int``: return a single gid if it belongs to the circuit.
                - ``sequence``: return gids in an array.
                - ``str``: return gids from a target.
                - ``mapping``: return gids matching a properties filter.
                - ``None``: return all gids.

                If ``group`` is a ``sequence``, the order of results is preserved.
                Otherwise the result is sorted and contains no duplicates.

            sample (int): If specified, randomly choose ``sample`` number of
                gids from the match result. If the size of the sample is greater than
                the size of the NodePopulation then all gids are taken and shuffled.

            limit (int): If specified, return the first ``limit`` number of
                gids from the match result. If limit is greater than the size of the population
                all node gids are returned.

        Returns:
            numpy.array: A numpy array of gids (1-based gids).

        Notes:
            Careful, even if the file is a sonata file, the GIDs used and returned by this
            function are 1-based GIDs only. This is done for historical reasons and compatibility
            with the nrn and target files.
        """
        if isinstance(group, MutableMapping):
            group_query = copy.deepcopy(group)
            # remove the target key and resolve it
            if '$target' in group_query:
                target = group_query.pop("$target")
                if target not in self._node_sets:
                    raise BluePyError(f"Target {target} doesn't exist")
                # Virtual gids are gids greater than self.size
                if np.min(self._node_sets[target]["node_id"]) > self.size and len(
                        group_query) > 0:
                    raise BluePyError("Could not apply properties filter to external target")
                group_query.update(self.get_node_set(target))
        else:
            group_query = group

        return self._resolve_ids_from_query(group=group_query, limit=limit, sample=sample)

    def get(self, group=None, properties=None):
        """Cell properties as pandas Series / DataFrame.
        Args:
            group (int/sequence/str/mapping/None): Which gids will have their properties returned
                depends on the type of the ``group`` argument:
                - ``int``: return the properties of a single gid.
                - ``sequence``: return the properties from a list of gids.
                - ``str``: return the properties of gids in a node set.
                - ``mapping``: return the properties of gids matching a properties filter.
                - ``None``: return the properties of all gids.

            properties (list): If specified, return only the properties in the list.
                Otherwise return all properties.

        Returns:
            value/pandas.Series/pandas.DataFrame:
                If single gid is passed as ``group`` and single property as properties it returns
                a single value. If single gid is passed as ``group`` and list as property
                returns a pandas Series. Otherwise return a pandas DataFrame indexed by
                gids (1-based gids).

        Notes:
            Careful, even if the file is a sonata file, the GIDs used and returned as indices by
            this function are 1-based GIDs only. This is done for historical reasons and
            compatibility with the nrn and target files.

            The returned DataFrame shouldn't be modified, to avoid affecting the original data.
        """
        result = self._data

        if group is not None:
            ids = self.ids(group=group)
            if np.count_nonzero(ids > self.size) > 0:
                raise BluePyError("Trying to access virtual gids properties.")
            if len(ids) == 1 and isinstance(group, (int, np.integer)):
                ids = ids[0]
            result = result.loc[ids]

        if properties is not None:
            for p in ensure_list(properties):
                self._check_property(p)
            result = result[properties]

        return result
