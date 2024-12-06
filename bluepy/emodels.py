""" EModel-related classes/methods. """
import os
from cached_property import cached_property

import pandas as pd

from bluepy.utils.url import get_file_path_by_url
from bluepy.enums import Cell


DEFAULT_EMCOMBO_KEYS = {'morph_name', 'layer', 'fullmtype', 'etype', 'emodel', 'combo_name'}


def _load_mecombo_info_from_file(mecombo_info_path):
    """Load the info from a mecombo_emodel.tsv file"""
    filepath = get_file_path_by_url(mecombo_info_path)
    return pd.read_csv(filepath, sep=r'\s+').set_index('combo_name', drop=False)


def _load_mecombo_info_from_circuit(circuit):
    """Load the mecombo from the circuit (will only contain a combo name)."""
    data = circuit.cells.get(None, Cell.ME_COMBO).unique()
    return pd.DataFrame(data, index=data, columns=['combo_name'])


class EModelHelper:
    """ Collection of emodel-related methods. """
    def __init__(self, url, circuit, mecombo_info=None):
        self._emodel_path = get_file_path_by_url(url)
        self._circuit = circuit
        if mecombo_info:
            self._mecombo_data = _load_mecombo_info_from_file(mecombo_info)
        else:
            self._mecombo_data = _load_mecombo_info_from_circuit(circuit)
        self._hoc_name_prop = 'emodel' if 'emodel' in self._mecombo_data else 'combo_name'

    @cached_property
    def _extra_properties(self):
        return set(self._mecombo_data) - DEFAULT_EMCOMBO_KEYS

    def _get_info_from_gid(self, gid, properties=None):
        me_combo = self._circuit.cells.get(gid, Cell.ME_COMBO)
        if not properties:
            return self._mecombo_data.loc[me_combo]
        return self._mecombo_data.loc[me_combo, properties]

    def get_mecombo_info(self, gid):
        """Access all emodel information for the corresponding `gid`."""
        return self._get_info_from_gid(gid).to_dict()

    def get_filepath(self, gid):
        """ Path to HOC emodel file corresponding to `gid`. """
        file_name = self._get_info_from_gid(gid, properties=self._hoc_name_prop) + ".hoc"
        return os.path.join(self._emodel_path, file_name)

    def get_properties(self, gid):
        """
        Dictionary with me_combo properties corresponding to `gid`.

        Returns None for old-style emodel releases with separate HOC for each me_combo.
        """
        if not self._extra_properties:
            return None
        return self._get_info_from_gid(gid, properties=list(self._extra_properties)).to_dict()
