""" Connectome access. """

from builtins import map  # pylint: disable=redefined-builtin

import os

from cached_property import cached_property

from bluepy.exceptions import require, BluePyError
from bluepy.utils import deprecate
from bluepy.utils.url import get_file_path_by_url
from bluepy.enums import Synapse


class Connectome:
    """ Connectome access. """

    def __init__(self, url, circuit, spatial_index=None, metadata=None):
        filepath = get_file_path_by_url(url)
        # explicitly raise for syn2 because it was confusing for users
        if filepath.endswith(".syn2"):
            raise BluePyError("'.syn2' format support has been dropped. Use '.sonata' instead.")

        # sonata files and nrn files can share the same extension '.h5'.
        try:
            from bluepy.impl.connectome_sonata import SonataConnectome
            self._impl = SonataConnectome(filepath, circuit)
        except (IOError, BluePyError):
            try:
                from bluepy.impl.connectome_nrn import NrnConnectome
                self._impl = NrnConnectome(filepath, circuit)
            except BluePyError:
                raise BluePyError(
                    "The connectome file cannot be read by bluepy. "
                    "Please use '.nrn' or latest 'sonata' format.")

        self._circuit = circuit
        self._index_url = spatial_index
        self.metadata = metadata

    @property
    def available_properties(self):
        """ Set of available Synapse properties. """
        return self._impl.available_properties

    def synapse_properties(self, synapse_ids, properties):
        """
        Synapse properties as pandas DataFrame.

        Args:
            synapse_ids: array-like of synapse IDs
            properties: `Synapse` property | list of `Synapse` properties

        Returns:
            Pandas Series indexed by synapse IDs if `properties` is scalar;
            Pandas DataFrame indexed by synapse IDs if `properties` is list.
        """
        if isinstance(properties, Synapse):
            # pylint: disable=maybe-no-member
            return self._impl.synapse_properties(synapse_ids, [properties]).iloc[:, 0]
        else:
            return self._impl.synapse_properties(synapse_ids, properties)

    def synapse_positions(self, synapse_ids, side, kind):
        """
        Synapse positions as pandas DataFrame.

        Args:
            synapse_ids: array-like of synapse IDs
            side: 'pre' | 'post'
            kind: 'center' | 'contour'

        Returns:
            Pandas Dataframe with ('x', 'y', 'z') columns indexed by synapse IDs.
        """
        props = {
            ('pre', 'center'): {
                Synapse.PRE_X_CENTER: 'x',
                Synapse.PRE_Y_CENTER: 'y',
                Synapse.PRE_Z_CENTER: 'z',
            },
            ('pre', 'contour'): {
                Synapse.PRE_X_CONTOUR: 'x',
                Synapse.PRE_Y_CONTOUR: 'y',
                Synapse.PRE_Z_CONTOUR: 'z',
            },
            ('post', 'center'): {
                Synapse.POST_X_CENTER: 'x',
                Synapse.POST_Y_CENTER: 'y',
                Synapse.POST_Z_CENTER: 'z',
            },
            ('post', 'contour'): {
                Synapse.POST_X_CONTOUR: 'x',
                Synapse.POST_Y_CONTOUR: 'y',
                Synapse.POST_Z_CONTOUR: 'z',
            },
        }[(side, kind)]

        result = self.synapse_properties(synapse_ids, list(props))
        result.rename(columns=props, inplace=True)
        result.sort_index(axis=1, inplace=True)

        return result

    def afferent_gids(self, gid):
        """
        Get afferent GIDs for given `gid`.

        Args:
            gid: postsynaptic GID

        Returns:
            Sorted array of unique GIDs.
        """
        return self._impl.afferent_gids(gid)

    def afferent_synapses(self, gid, properties=None):
        """
        Get afferent synapses for given `gid`.

        Args:
            gid: postsynaptic GID
            properties: None / `Synapse` property / list of `Synapse` properties

        Returns:
            List of synapse IDs, if `properties` is None;
            Pandas Series indexed by synapse IDs if `properties` is scalar;
            Pandas DataFrame indexed by synapse IDs if `properties` is list.
        """
        return self.pathway_synapses(None, [gid], properties)

    def efferent_gids(self, gid):
        """
        Get efferent GIDs for given `gid`.

        Args:
            gid: presynaptic GID

        Returns:
            Sorted array of unique GIDs.
        """
        return self._impl.efferent_gids(gid)

    def efferent_synapses(self, gid, properties=None):
        """
        Get efferent synapses for given `gid`.

        Args:
            gid: presynaptic GID
            properties: None / `Synapse` property / list of `Synapse` properties

        Returns:
            List of synapse IDs, if `properties` is None;
            Pandas Series indexed by synapse IDs if `properties` is scalar;
            Pandas DataFrame indexed by synapse IDs if `properties` is list.
        """
        return self.pathway_synapses([gid], None, properties)

    def pair_synapses(self, pre_gid, post_gid, properties=None):
        """
        Get synapses corresponding to `pre_gid` -> `post_gid` connection.

        Args:
            pre_gid: presynaptic GID
            post_gid: postsynaptic GID
            properties: None / `Synapse` property / list of `Synapse` properties

        Returns:
            List of synapse IDs, if `properties` is None;
            Pandas Series indexed by synapse IDs if `properties` is scalar;
            Pandas DataFrame indexed by synapse IDs if `properties` is list.
        """
        return self.pathway_synapses([pre_gid], [post_gid], properties)

    def pathway_synapses(self, pre=None, post=None, properties=None):
        """
        Get synapses corresponding to `pre` -> `post` connections.

        Args:
            pre: presynaptic cell group
            post: postsynaptic cell group
            properties: None / `Synapse` property / list of `Synapse` properties

        Returns:
            List of synapse IDs, if `properties` is None;
            Pandas Series indexed by synapse IDs if `properties` is scalar;
            Pandas DataFrame indexed by synapse IDs if `properties` is list.
        """
        pre_gids = self._resolve_gids(pre)
        post_gids = self._resolve_gids(post)

        if isinstance(properties, Synapse):
            return self._impl.pathway_synapses(pre_gids, post_gids, [properties]).iloc[:, 0]
        else:
            return self._impl.pathway_synapses(pre_gids, post_gids, properties)

    def iter_connections(
        self, pre=None, post=None, unique_gids=False, shuffle=False,
        return_synapse_ids=False, return_synapse_count=False
    ):
        """
        Iterate through `pre` -> `post` connections.

        Args:
            pre: presynaptic cell group
            post: postsynaptic cell group
            unique_gids: if True, no gid will be used more than once as pre and post synaptic
                gids. Careful, this flag does not provide unique (pre, post) pairs but unique gids.
            shuffle: if True, result order would be (somewhat) randomised
            return_synapse_count: if True, synapse count is added to yield result
            return_synapse_ids: if True, synapse ID list is added to yield result

        `return_synapse_count` and `return_synapse_ids` are mutually exclusive.

        Yields:
            (pre_gid, post_gids, synapse_ids) if return_synapse_ids == True;
            (pre_gid, post_gid, synapse_count) if return_synapse_count == True;
            (pre_gid, post_gid) otherwise.
        """
        if return_synapse_ids and return_synapse_count:
            raise BluePyError(
                "`return_synapse_count` and `return_synapse_ids` are mutually exclusive"
            )

        pre_gids = self._resolve_gids(pre)
        post_gids = self._resolve_gids(post)

        it = self._impl.iter_connections(pre_gids, post_gids, unique_gids, shuffle)

        if return_synapse_count:
            return it
        elif return_synapse_ids:
            def add_synapse_ids(x):
                return x[0], x[1], self.pair_synapses(x[0], x[1])
            return map(add_synapse_ids, it)
        else:
            def omit_synapse_count(x):
                return x[:2]
            return map(omit_synapse_count, it)

    @cached_property
    def spatial_index(self):
        """ Spatial index. """
        from bluepy.impl.connectome_sonata import SonataConnectome
        from bluepy.index.indices import SynapseIndex

        if isinstance(self._impl, SonataConnectome):
            deprecate.fail('Not supported for Sonata edges due to [BLPY-238]')
        require(self._index_url is not None, "Spatial index not defined")
        return SynapseIndex(os.path.join(get_file_path_by_url(self._index_url), "SYNAPSE"))

    def _resolve_gids(self, group):
        """ GIDs corresponding to cell group. """
        return None if group is None else self._circuit.cells.ids(group)
