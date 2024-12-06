""" Access to circuit data. """
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from cached_property import cached_property

from bluepy_configfile.configfile import BlueConfig

from bluepy.utils import open_utf8
from bluepy.utils.url import get_file_path_by_url, is_local_path
from bluepy.exceptions import BluePyError

from bluepy.cells import CellCollection
from bluepy.connectome import Connectome
from bluepy.emodels import EModelHelper
from bluepy.morphology import MorphHelper
from bluepy.circuit_stats import StatsHelper
from bluepy import settings
from bluepy.paths import _find_circuit_anchor


def _find_cells(blueconfig):
    """ Find MVD3 / MVD2 / Node file used by BlueConfig. """
    circuit_path = blueconfig.Run.CircuitPath
    filenames = ['circuit.mvd3', 'circuit.mvd2']
    filepaths = [os.path.join(circuit_path, f) for f in filenames]

    try:
        cell_path = blueconfig.Run.CellLibraryFile
        if cell_path not in filenames:
            filepaths = [blueconfig.Run.CellLibraryFile] + filepaths
    except AttributeError:
        pass

    for filepath in filepaths:
        if os.path.exists(filepath):
            result = filepath
            break
    else:
        raise BluePyError(f"No Cell file found at {circuit_path}")
    return result


def _find_targets(blueconfig):
    """ Find '.target' file(s) used by BlueConfig. """
    result = []
    for root_dir in [blueconfig.Run.nrnPath, blueconfig.Run.CircuitPath]:
        filepath = os.path.join(root_dir, 'start.target')
        if os.path.exists(filepath):
            result.append(filepath)
            break
    else:
        raise BluePyError(f"No 'start.target' file found at {blueconfig.Run.CircuitPath}")
    if 'TargetFile' in blueconfig.Run:
        filename = blueconfig.Run.TargetFile
        if os.path.isabs(filename):
            filepath = filename
        else:
            filepath = os.path.join(_find_circuit_anchor(blueconfig), filename)
        if os.path.exists(filepath):
            result.append(filepath)
        else:
            raise BluePyError(f"'{filepath}' file not found")
    return result


def _find_spatial_index(dirpath, prefix):
    for filename in ['_index.dat', '_index.idx', '_payload.dat']:
        if not os.path.exists(os.path.join(dirpath, prefix + filename)):
            return None
    return dirpath


def _find_segment_index(blueconfig):
    return _find_spatial_index(blueconfig.Run.CircuitPath, 'SEGMENT')


def _find_synapse_index(blueconfig):
    dirpath = blueconfig.Run.nrnPath
    if os.path.isfile(dirpath):
        dirpath = os.path.dirname(dirpath)
    return _find_spatial_index(dirpath, 'SYNAPSE')


def _parse_blueconfig(blueconfig):
    """ Parse circuit config from BlueConfig object. """
    result = {}

    result['cells'] = _find_cells(blueconfig)

    result['morphologies'] = blueconfig.Run.MorphologyPath
    if 'MorphologyType' in blueconfig.Run:
        result['morphology_type'] = blueconfig.Run.MorphologyType

    result['emodels'] = blueconfig.Run.METypePath
    if 'MEComboInfoFile' in blueconfig.Run:
        result['mecombo_info'] = blueconfig.Run.MEComboInfoFile

    result['connectome'] = blueconfig.Run.nrnPath

    result['targets'] = _find_targets(blueconfig)

    result['projections'] = {
        s.name: s.Path for s in blueconfig.typed_sections('Projection')
    }

    # we have to call explicitly the s[k] to call the getitem in python3
    # if we don t, then a ConfValue is return in python3 and a ConfValue.value in python2.
    result['projections_metadata'] = {
        s.name: {k: s[k] for k in s} for s in blueconfig.typed_sections('Projection')
    }

    result['segment_index'] = _find_segment_index(blueconfig)
    result['synapse_index'] = _find_synapse_index(blueconfig)

    if 'Atlas' in blueconfig.Run:
        result['atlas'] = blueconfig.Run.Atlas

    subcellular_path = os.path.join(blueconfig.Run.CircuitPath, 'subcellular.h5')
    if os.path.exists(subcellular_path):
        result['subcellular'] = subcellular_path

    return result


def _resolve_paths(config, circuit_dir):
    """ Resolve relative paths used in circuit config (i.e. those starting with './'). """

    def _visit(value):
        if isinstance(value, str):
            if value.startswith("./"):
                value = os.path.abspath(os.path.join(circuit_dir, value))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                value[k] = _visit(v)
        elif isinstance(value, Sequence):
            for i, v in enumerate(value):
                value[i] = _visit(v)
        return value

    _visit(config)


def _get_circuit_config(url):
    """ Get circuit config dictionary from URL. """
    if is_local_path(url):
        filepath = get_file_path_by_url(url)
        with open_utf8(filepath) as f:
            config_dir = os.path.dirname(url)
            if filepath.endswith(".json"):
                result = json.load(f)
                _resolve_paths(result, circuit_dir=config_dir)
            else:
                result = _parse_blueconfig(BlueConfig(f))
    else:
        raise NotImplementedError("Entity management has been removed. Use local path instead.")

    return result


def _load_config(config):
    """ load the config """
    if isinstance(config, BlueConfig):
        return _parse_blueconfig(config)
    elif isinstance(config, (str, Path, )):
        return _get_circuit_config(str(config))
    elif isinstance(config, Mapping):
        return config
    else:
        raise BluePyError(f"Unexpected config type: {type(config)}")


class Circuit:
    """ Access to circuit data. """

    def __init__(self, config):
        self._config = _load_config(config)
        self._projections = {}

    @property
    def config(self):
        """ Access to the config object. """
        return self._config

    @cached_property
    def cells(self):
        """ Access to cell properties. """
        return CellCollection(
            self._config['cells'],
            targets=self._config.get('targets'),
            spatial_index=self._config.get('soma_index')
        )

    @cached_property
    def connectome(self):
        """ Access to main connectome. """
        return Connectome(
            self._config['connectome'], self,
            spatial_index=self._config.get('synapse_index')
        )

    def projection(self, projection):
        """ Access to projection connectome. """
        if projection in self._projections:
            result = self._projections[projection]
        else:
            path = self._config['projections'][projection]
            try:
                metadata = self._config['projections_metadata'][projection]
            except KeyError:
                metadata = None
            result = Connectome(path, self, metadata=metadata)
            self._projections[projection] = result
        return result

    @cached_property
    def morph(self):
        """ Access to cell morphologies. """
        # apply the morph_type here to avoid adding a mandatory field breaking the dict constructors
        morph_type = self._config.get('morphology_type', "h5v1")
        return MorphHelper(
            self._config['morphologies'], self,
            spatial_index=self._config.get('segment_index'),
            morph_type=morph_type
        )

    @cached_property
    def emodels(self):
        """ Access to cell emodels. """
        return EModelHelper(
            self._config['emodels'], self,
            mecombo_info=self._config.get('mecombo_info')
        )

    @cached_property
    def subcellular(self):
        """ Access to subcellular data. """
        from bluepy.subcellular import SubcellularHelper
        return SubcellularHelper(
            self._config['subcellular'], self
        )

    @cached_property
    def stats(self):
        """ Collection of circuit stats utility methods. """
        return StatsHelper(self)

    @cached_property
    def atlas(self):
        """ Access to atlas used for building the circuit. """
        from voxcell.nexus.voxelbrain import Atlas
        return Atlas.open(self._config['atlas'], cache_dir=settings.ATLAS_CACHE_DIR)

    def __getstate__(self):
        """ make Circuits pickle-able, without storing state of caches"""
        return self.config

    def __setstate__(self, state):
        """ load from pickle state """
        self.__init__(state)
