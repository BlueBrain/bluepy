import os
import pickle
import warnings
from unittest.mock import Mock, patch

import pytest
from bluepy_configfile.configfile import BlueConfigFile
from utils import tmp_file

import bluepy.circuit
import bluepy.simulation as test_module
from bluepy.exceptions import BluePyError
from bluepy.impl import compartment_report, spike_report

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def _blueconfig():
    bc = BlueConfigFile()
    bc.add_section('Run', 'Default', '')
    bc.Run.CircuitTarget = 'foo'
    bc.Run.OutputRoot = TEST_DATA_DIR
    bc.Run.Duration = 100
    bc.add_section('Report', 'rep1', '')
    bc.Report_rep1.Type = 'synapse'
    bc.Report_rep1.Format = 'SONATA'
    bc.add_section('Report', 'rep2', '')
    bc.Report_rep2.Type = 'synapse'
    bc.Report_rep2.Format = 'Bin'
    bc.add_section('Report', 'rep3', '')
    bc.Report_rep3.Type = 'synapse'
    bc.Report_rep3.Format = 'Unknown'
    return bc


class TestSimulation:
    def setup_method(self):
        self.sim = test_module.Simulation(_blueconfig())

    def test_meta(self):
        assert self.sim.t_start == 0.0
        assert self.sim.t_end == 100.

    def test_target_1(self):
        assert self.sim.target == 'foo'

    def test_target_2(self):
        del self.sim.config.Run['CircuitTarget']
        assert self.sim.target is None

    @patch.object(test_module.Simulation, 'circuit')
    def test_target_gids(self, circuit_mock):
        self.sim.target_gids
        circuit_mock.cells.ids.assert_called_with('foo')

    @patch.object(bluepy.circuit, '_parse_blueconfig')
    def test_circuit(self, _):
        assert isinstance(self.sim.circuit, bluepy.circuit.Circuit)

    @patch("os.path.exists", return_value=True)
    def test_spikes(self, mock):
        assert isinstance(self.sim.spikes, spike_report.SpikeReport)

    @patch("os.path.exists", return_value=True)
    @patch.object(bluepy.circuit, '_parse_blueconfig')
    @patch.object(compartment_report.TimeSeriesReport, '_open')
    def test_report(self, _1, _2, _3):
        assert isinstance(self.sim.report('rep1'), compartment_report.SynapseReport)
        assert isinstance(self.sim.report('rep2'), compartment_report.SynapseReport)

        pytest.raises(BluePyError, self.sim.report, 'err')
        pytest.raises(BluePyError, self.sim.report, 'rep3')

    def test_pickle_roundtrip(self):
        dumped = pickle.dumps(self.sim)
        loaded = pickle.loads(dumped)

        assert isinstance(loaded.config, BlueConfigFile)
        assert loaded.config == self.sim.config


def test_read_blueconfig_1():
    bc = _blueconfig()
    assert bc == test_module._read_blueconfig(bc)


def test_read_blueconfig_2():
    filepath = os.path.join(TEST_DATA_DIR, "CircuitConfig")
    assert isinstance(test_module._read_blueconfig(filepath), BlueConfigFile)


def test_read_blueconfig_raises():
    with pytest.raises(BluePyError):
        test_module._read_blueconfig(42)


def test_deduce_report_type_1():
    circuit = Mock()
    circuit.cells._targets._targets = {
        'All': Mock(type="Cell"),
        'AllComp': Mock(type="Compartment"),
    }
    assert test_module._deduce_report_type({'Type': 'compartment', 'Target': 'All'},
                                           circuit) == compartment_report.SomaReport
    assert test_module._deduce_report_type({'Type': 'summation', 'Target': 'All'},
                                           circuit) == compartment_report.SomaReport


def test_deduce_report_type_2():
    circuit = Mock()
    circuit.cells._targets._targets = {
        'All': Mock(type="Cell"),
        'AllComp': Mock(type="Compartment"),
        'AllSection': Mock(type="Section"),
        'UnknownType': Mock(type="Unknown"),
    }
    assert test_module._deduce_report_type({'Type': 'compartment', 'Target': 'AllComp'},
                                           circuit) == compartment_report.CompartmentReport
    assert test_module._deduce_report_type({'Type': 'summation', 'Target': 'AllComp'},
                                           circuit) == compartment_report.CompartmentReport
    assert test_module._deduce_report_type({'Type': 'compartment', 'Target': 'AllSection'},
                                           circuit) == compartment_report.CompartmentReport
    assert test_module._deduce_report_type({'Type': 'summation', 'Target': 'AllSection'},
                                           circuit) == compartment_report.CompartmentReport

    with pytest.raises(BluePyError):
        test_module._deduce_report_type({'Type': 'compartment', 'Target': 'UnknownType'}, circuit)


def test_deduce_report_type_3():
    assert test_module._deduce_report_type({'Type': 'synapse'}, None) == compartment_report.SynapseReport
    assert test_module._deduce_report_type({'Type': 'Synapse'}, None) == compartment_report.SynapseReport


def test_deduce_report_type_raises_1():
    with pytest.raises(BluePyError):
        test_module._deduce_report_type({'Type': 'foo'}, None)


def test_deduce_report_type_raises_2():
    circuit = Mock()
    circuit.cells._targets._targets = {
        'Foo': Mock(type="Foo"),
    }
    with pytest.raises(BluePyError):
        test_module._deduce_report_type({'Type': 'compartment', 'Target': 'Foo'}, circuit)


@patch("os.path.exists", return_value=True)
def test_outputroot_path(mock):
    bc = BlueConfigFile()
    bc.add_section('Run', 'Default', '')
    bc.Run.CurrentDir = str(TEST_DATA_DIR)
    bc.Run.OutputRoot = 'out'
    assert test_module.PathHelpers.output_dir(bc) == os.path.join(TEST_DATA_DIR, 'out')

    bc = BlueConfigFile()
    bc.add_section('Run', 'Default', '')
    bc.Run.CurrentDir = str(TEST_DATA_DIR)
    bc.Run.OutputRoot = '/out'
    assert test_module.PathHelpers.output_dir(bc) == '/out'

    bc = BlueConfigFile(open(os.path.join(TEST_DATA_DIR, 'CircuitConfig')))
    assert test_module.PathHelpers.output_dir(bc) == os.path.join(TEST_DATA_DIR, 'foobar')

    with pytest.raises(BluePyError):
        bc = BlueConfigFile()
        bc.add_section('Run', 'Default', '')
        bc.Run.OutputRoot = 'out'
        test_module.PathHelpers.output_dir(bc)


def test_outputroot_path_exists_fails():
    with pytest.raises(BluePyError):
        bc = BlueConfigFile(open(os.path.join(TEST_DATA_DIR, 'CircuitConfig')))
        assert test_module.PathHelpers.output_dir(bc) == os.path.join(TEST_DATA_DIR, 'foobar')

    with pytest.raises(BluePyError):
        bc = BlueConfigFile()
        bc.add_section('Run', 'Default', '')
        bc.Run.OutputRoot = '/unknown'
        test_module.PathHelpers.output_dir(bc)


def test_report_sources():
    content = """
Run Default
{{
    CircuitPath {dir}
    nrnPath {dir}
    CellLibraryFile circuit.mvd3
    METypePath {dir}
    MorphologyPath {dir}
    TargetFile {dir}/{target}
    OutputRoot {dir}

    CircuitTarget A
    Duration 1000
    Dt 0.025
    ForwardSkip 5000
}}

Report soma_sonata
{{
    Target A
    Type compartment
    Format SONATA
    ReportOn v
    Unit mV
    Dt 0.1
    StartTime 0
    EndTime 1000
}}

Report soma
{{
    Target A
    Type compartment
    Format Bin
    ReportOn v
    Unit mV
    Dt 0.1
    StartTime 0
    EndTime 1000
}}

Report soma_h5
{{
    Target A
    Type compartment
    Format HDF5
    ReportOn v
    Unit mV
    Dt 0.1
    StartTime 0
    EndTime 1000
}}
}}
""".format(dir=TEST_DATA_DIR, target="E.target")
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        tested = test_module.Simulation(filepath)
        assert tested.report_names == {'soma', 'soma_sonata', 'soma_h5'}
        sonata_report = tested.report("soma_sonata")
        assert type(sonata_report._reader) == compartment_report.LibSonataReader
        bbp_soma_report = tested.report("soma")
        assert type(bbp_soma_report._reader) == compartment_report.BrionReport
        h5_soma_report = tested.report("soma_h5")
        assert type(h5_soma_report._reader) == compartment_report.BrionReport
        with warnings.catch_warnings(record=True) as w:
            tested.report("soma", source="h5")
            assert len(w) == 1
            assert "BlueConfig" in str(w[-1].message)


def test_report_sources_fail():
    content = """
Run Default
{{
    CircuitPath {dir}
    nrnPath {dir}
    CellLibraryFile circuit.mvd3
    METypePath {dir}
    MorphologyPath {dir}
    TargetFile {dir}/{target}
    OutputRoot {dir}/nrn

    CircuitTarget A
    Duration 1000
    Dt 0.025
    ForwardSkip 5000
}}

Report not_exists
{{
    Target A
    Type compartment
    Format SONATA
    ReportOn v
    Unit mV
    Dt 0.1
    StartTime 0
    EndTime 1000
}}
}}
    """.format(dir=TEST_DATA_DIR, target="E.target")
    with tmp_file(TEST_DATA_DIR, content, cleanup=True) as filepath:
        blueconfig = test_module._read_blueconfig(filepath)
        with pytest.raises(BluePyError) as e:
            test_module.PathHelpers.binreport_path(blueconfig, "not_exists", "h5")
        assert "Cannot find the report named" in str(e.value)

        with pytest.raises(BluePyError) as e2:
            test_module.PathHelpers.spike_report_path(blueconfig)
        assert "Cannot find the spike report" in str(e2.value)
