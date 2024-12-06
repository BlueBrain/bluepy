""" Access to simulation data. """

import os
import logging
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from cached_property import cached_property

from bluepy_configfile.configfile import BlueConfig

from bluepy.exceptions import BluePyError

from bluepy.paths import _find_circuit_anchor
from bluepy.utils import deprecate, open_utf8
from bluepy.impl.spike_report import SpikeReport
from bluepy.impl.compartment_report import CompartmentReport, SomaReport, SynapseReport

L = logging.getLogger(__name__)


FORMAT_EXT_DISPATCH = {"Bin": "bbp", "HDF5": "h5", "SONATA": "h5"}


class PathHelpers:
    """ Paths to simulation files. """

    @staticmethod
    def output_dir(config):
        """Path to simulation output folder."""
        output_root = config.Run.OutputRoot
        if os.path.isabs(output_root):
            output_path = output_root
        else:
            output_path = os.path.join(_find_circuit_anchor(config), output_root)
        if os.path.exists(output_path):
            return output_path
        raise BluePyError(f'Trying to access the OutputRoot : {output_path}, '
                          'but it does not exist.')

    @staticmethod
    def spike_report_path(config):
        """Path to spike report.

        There is no Format key for the spike reports so we do a lookup with first .dat and then .h5.
        """
        output_dir = PathHelpers.output_dir(config)
        for ext in ['dat', 'h5']:
            filepath = os.path.join(output_dir, f'out.{ext}')
            if os.path.exists(filepath):
                return filepath
        raise BluePyError("Cannot find the spike report.")

    @staticmethod
    def binreport_path(config, report_name, source):
        """Path to `report_name` binary report.

        Args:
            config(BlueConfig): the simulation config.
            report_name(str): the report name.
            source(str): the report source (should be 'h5' or 'bbp').
        """
        filename = f"{report_name}.{source}"
        filepath = os.path.join(PathHelpers.output_dir(config), filename)
        if os.path.exists(filepath):
            return filepath
        raise BluePyError(f"Cannot find the report named {report_name} for source : {source}")


def _read_blueconfig(obj):
    if isinstance(obj, BlueConfig):
        return obj
    elif isinstance(obj, Mapping):
        return BlueConfig(obj)
    elif isinstance(obj, (str, Path)):
        with open_utf8(obj) as f:
            return BlueConfig(f)
    else:
        raise BluePyError(f"Unexpected config type: {type(obj)}")


def _get_target_type(target, circuit):
    """ Get target type (Cell / Compartment) for the given target name. """
    # pylint: disable=protected-access
    # In forthcoming circuit representation (SONATA), node group definitions
    # will not bear additional information about report target sections;
    # instead, this would be a part of the report config.
    # Thus we tolerate querying this information in such an ugly way as a temporary measure.
    return circuit.cells._targets._targets[target].type


def _deduce_report_type(report_config, circuit):
    """ Deduce report type from its config. """
    report_type = report_config['Type'].lower()
    if report_type in ['compartment', 'summation']:
        target_type = _get_target_type(report_config['Target'], circuit).lower()
        if target_type == 'cell':
            return SomaReport
        elif target_type in ['compartment', 'section']:
            return CompartmentReport
        else:
            raise BluePyError(f"Unknown target type '{target_type}'")
    elif report_type == 'synapse':
        return SynapseReport
    else:
        raise BluePyError(f"Unknown report type '{report_type}'")


class SimulationPlotHelper:  # pragma: no cover
    """ Simulation plot helper. """

    def __init__(self, sim):
        self._sim = sim

        def _bind_plotting_functions():
            """ Bind the plotting functions to the SimulationPlotHelper instance """
            from bluepy import plotting
            for func_name in plotting.SIMULATION_PLOTS:
                setattr(self, func_name, self._set_sim(self._sim, getattr(plotting, func_name)))

        _bind_plotting_functions()

    @staticmethod
    def _set_sim(sim, func):
        """ Make functions from bluepy.plotting directly usable by SimulationPlotHelper

        Args:
            sim: a simulation object
            func: a function (from the plotting module)

        Returns:
            The wrapped function func with sim set as the first argument
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """ The wrapped function with the correct number of argument and sim as first """
            return func(sim, *args, **kwargs)

        return wrapper


class Simulation:
    """ Access to simulation data. """

    def __init__(self, config):
        self.config = _read_blueconfig(config)
        self._reports = {}

    @cached_property
    def circuit(self):
        """ Simulated circuit. """
        from bluepy.circuit import Circuit
        return Circuit(self.config)

    @property
    def t_start(self):
        """ Simulation start time. """
        return 0.0

    @property
    def t_end(self):
        """ Simulation end time. """
        return float(self.config.Run.Duration)

    @property
    def target(self):
        """ Simulation target, if specified (else None). """
        if 'CircuitTarget' in self.config.Run:
            return self.config.Run.CircuitTarget
        else:
            return None

    @property
    def target_gids(self):
        """ Simulation target GIDs. """
        return self.circuit.cells.ids(self.target)

    @property
    def report_names(self):
        """Returns the report names."""
        return set(self.config.to_dict()['Report'].keys())

    def report(self, name, source=None):
        """Binary report."""
        if source:
            deprecate.warn("Since bluepy>=2.4.0, the `source` argument in simulation.report is "
                           "ignored and the BlueConfig is used to detect the report format "
                           "instead.")
        result = self._reports.get(name)
        if result is None:
            reports = self.config.to_dict()['Report']
            if name not in reports:
                raise BluePyError(f"`name` should be one of [{','.join(reports)}]")
            report_config = reports[name]
            report_type = _deduce_report_type(report_config, self.circuit)
            source = FORMAT_EXT_DISPATCH.get(report_config["Format"])
            if source is None:
                raise BluePyError(f"Wrong Format in BlueConfig. Should be Bin, HDF5 or "
                                  f"SONATA. Found : {report_config['Format']}.")
            report_path = PathHelpers.binreport_path(self.config, name, source)
            result = report_type(report_path)
            self._reports[name] = result
        return result

    @cached_property
    def spikes(self):
        """ Spike report. """
        filepath = PathHelpers.spike_report_path(self.config)
        return SpikeReport.load(filepath)

    @cached_property
    def plot(self):  # pragma: no cover
        """ Simulation plot helper. """
        return SimulationPlotHelper(self)

    def __getstate__(self):
        """ make Simulations pickle-able, without storing state of caches"""
        return self.config

    def __setstate__(self, state):
        """ load from pickle state """
        self.__init__(state)
