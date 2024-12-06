import os

import pytest
from bluepy_configfile.configfile import BlueConfigFile

import bluepy.paths as test_module
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
TEST_BLUECONFIG = os.path.join(TEST_DATA_DIR, "CircuitConfig")


def test__find_circuit_anchor():
    bc = BlueConfigFile()
    bc.add_section('Run', 'Default', '')
    bc.Run.CurrentDir = "/current_dir"
    assert test_module._find_circuit_anchor(bc) == "/current_dir"

    # use the blueconfig parent directory
    bc = BlueConfigFile(open(TEST_BLUECONFIG))
    bc.add_section('Run', 'Default', '')
    assert test_module._find_circuit_anchor(bc) == TEST_DATA_DIR

    # blueconfig created programmatically --> no file --> no parent directory
    with pytest.raises(BluePyError):
        bc = BlueConfigFile()
        bc.add_section('Run', 'Default', '')
        bc.Run.CurrentDir = "./current_dir"
        test_module._find_circuit_anchor(bc)

    # blueconfig created programmatically --> no file --> no parent directory
    with pytest.raises(BluePyError):
        bc = BlueConfigFile()
        bc.add_section('Run', 'Default', '')
        test_module._find_circuit_anchor(bc)
