""" Access to subcellular data. """

import pandas as pd
import tables

from cached_property import cached_property

from bluepy.enums import Synapse
from bluepy.utils.url import get_file_path_by_url


class Organelle:
    """ Organelle names. """
    TOTAL = 'total'
    NUCLEUS = 'nucleus'
    CYTOSOL = 'cytosol'
    ER = 'ER'
    ENDOSOME = 'endosome'
    GOLGI = 'golgi'
    LYSOSOME = 'lysosome'
    MITOCHONDRION = 'mitochondrion'
    PEROXISOME = 'peroxisome'
    MEMBRANE = 'membrane'


class SubcellularHelper:
    """ Access to subcellular data. """

    def __init__(self, url, circuit):
        self._filepath = get_file_path_by_url(url)
        self._circuit = circuit

    def _read_table(self, node):
        with tables.open_file(self._filepath, 'r') as h5f:
            result = pd.DataFrame(h5f.get_node(node).read())
        # convert all string columns from 'bytes' to 'text'
        for c in result.select_dtypes(object):
            result[c] = result[c].str.decode('ascii')
        return result

    def _gid_data(self, gid, data):
        uuid = self._gid_mapping.loc[gid, data]
        node = f"/library/{data}/{uuid}"
        return self._read_table(node)

    @cached_property
    def _gid_mapping(self):
        return self._read_table('/cells').set_index('gid')

    def gene_mapping(self, genes=None):
        """
        Gene to protein correspondence.

        Args:
            genes: gene(s) to query (``None`` for all)

        Example:
            >> c.v2.subcellular.gene_mapping(['gene1', 'gene2'])

        Returns:
            pandas.DataFrame with protein IDs, indexed by gene names
        """
        data = self._read_table('/library/gene_mapping')
        data.set_index('gene', inplace=True)
        if genes is not None:
            data = data.loc[genes]
        return data

    def gene_expressions(self, gid, genes=None):
        """
        Gene expressions for given GID.

        Args:
            gid: GID of interest
            genes: gene(s) to query (``None`` for all)

        Returns:
            pandas.Series indexed by gene names

        Example:
            >> c.v2.subcellular.gene_expressions(42, genes=['gene1', 'gene2'])
        """
        data = self._gid_data(gid, 'gene_expressions').set_index('gene').iloc[:, 0]
        if genes is not None:
            data = data.loc[genes]
        return data

    def cell_proteins(self, gid, organelles=None, genes=None):
        """
        Protein concentration in organelles for given GID.

        Args:
            gid: GID of interest
            organelles: :class:`Organelle` name(s) (``None`` for all)
            genes: gene(s) to query (``None`` for all)

        Returns:
            pandas.DataFrame with protein concentrations (nM);
            rows correspond to gene names; columns -- to organelles

        Example:
            >> c.v2.subcellular.cell_proteins(42, [Organelle.NUCLEUS], genes='Zzz3')
        """
        data = self._gid_data(gid, 'cell_proteins')
        data.set_index('gene', inplace=True)
        if organelles is not None:
            data = data[organelles]
        if genes is not None:
            data = data.loc[genes]
        return data

    def synapse_proteins(self, synapse_id, side, genes=None, area_per_nS=None):
        """
        Protein concentrations for a given synapse.

        Args:
            synapse_id: synapse ID of interest
            side: 'pre' | 'post'
            genes: gene(s) to query (``None`` for all)
            area_per_nS: scaling factor to estimate synapse area (um^2) from conductance (nS)

        Returns:
            pandas.Series with protein concentrations indexed by gene names;
            measured in *counts* for postsynaptic side; and nM for presynaptic side.

        Example:
            >> c.v2.subcellular.synapse_proteins((42, 1), 'post')
        """
        if side not in ('pre', 'post'):
            raise ValueError("side should be either 'pre' or 'post'")
        syn = self._circuit.connectome.synapse_properties(
            [synapse_id], [Synapse.POST_GID, Synapse.G_SYNX, Synapse.TYPE]
        ).iloc[0]
        data = self._gid_data(syn[Synapse.POST_GID], 'synapse_proteins').set_index('gene')
        if side == 'pre':
            data = data['pre']  # pylint: disable=unsubscriptable-object
        else:
            is_excitatory = syn[Synapse.TYPE] >= 100
            if area_per_nS is None:
                area_per_nS = 0.12 if is_excitatory else 0.071
            col = 'post_exc' if is_excitatory else 'post_inh'
            # pylint: disable=unsubscriptable-object
            data = area_per_nS * syn[Synapse.G_SYNX] * data[col]
        if genes is not None:
            data = data.loc[genes]
        return data
