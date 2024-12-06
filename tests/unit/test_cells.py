import os

import pytest

import bluepy.cells as test_module
from bluepy.exceptions import BluePyError
from bluepy.impl.cells_mvd import MVDCellCollection
from bluepy.impl.cells_sonata import SonataCellCollection

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_cell():
    cell_mvd2_path = os.path.join(TEST_DATA_DIR, "circuit.mvd2")
    cell_mvd3_path = os.path.join(TEST_DATA_DIR, "circuit.mvd3")
    cell_sonata_path = os.path.join(TEST_DATA_DIR, "circuit.sonata")
    cell_sonata_h5_path = os.path.join(TEST_DATA_DIR, "circuit.h5")
    cell_unknown_path = os.path.join(TEST_DATA_DIR, "out.dat")

    cells_mvd2 = test_module.CellCollection(cell_mvd2_path)
    assert isinstance(cells_mvd2._impl, MVDCellCollection)
    assert cells_mvd2.mtypes == {'L6_Y', 'L2_X'}
    assert cells_mvd2.etypes == {'bNA', 'cNA'}
    assert cells_mvd2.count() == 3

    cells_mvd3 = test_module.CellCollection(cell_mvd3_path)
    assert isinstance(cells_mvd3._impl, MVDCellCollection)
    assert cells_mvd3.mtypes == {'L6_Y', 'L2_X'}
    assert cells_mvd3.etypes == {'bNA', 'cNA'}
    assert cells_mvd3.count() == 3

    cells_sonata = test_module.CellCollection(cell_sonata_path)
    assert isinstance(cells_sonata._impl, SonataCellCollection)
    assert cells_sonata.mtypes == {'L6_Y', 'L2_X'}
    assert cells_sonata.etypes == {'bNA', 'cNA'}
    assert cells_sonata.count() == 3

    cells_sonata_h5 = test_module.CellCollection(cell_sonata_h5_path)
    assert isinstance(cells_sonata._impl, SonataCellCollection)
    assert cells_sonata_h5.mtypes == {'L6_Y', 'L2_X'}
    assert cells_sonata_h5.etypes == {'bNA', 'cNA'}
    assert cells_sonata_h5.count() == 3
    with pytest.raises(BluePyError):
        test_module.CellCollection(cell_unknown_path)


def test_available_properties():
    cell_mvd3_path = os.path.join(TEST_DATA_DIR, "circuit.mvd3")
    cells_mvd3 = test_module.CellCollection(cell_mvd3_path)
    res = cells_mvd3.available_properties
    expected = {'minicolumn', 'synapse_class', 'layer', 'x', 'morphology', 'mtype',
                'orientation', 'y', 'morph_class', 'z', 'etype', 'me_combo', 'hypercolumn'}
    missing = res - expected
    assert len(missing) == 0
