import os
from unittest.mock import Mock

import pandas as pd

import bluepy.emodels as test_module

TEST_EMODEL_DIR = "foo"

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
TEST_MECOMBO_INFO = os.path.join(TEST_DATA_DIR, "mecombo_emodel.tsv")


def _mock_circuit_get(gid, prop):
    values = {1: "mecombo-1", 2: "mecombo-2"}
    if prop != "me_combo":
        raise KeyError(f"The mocked get in tests miss the {prop} property")
    if gid is None:
        return pd.Series(list(values.values()))
    return values[gid]


class TestEModelHelper:
    def setup_method(self):
        circuit = Mock()
        circuit.cells.get = _mock_circuit_get
        self.emodels = test_module.EModelHelper(
            TEST_EMODEL_DIR, circuit, mecombo_info=TEST_MECOMBO_INFO
        )

    def test_get_filepath(self):
        actual = self.emodels.get_filepath(1)
        expected = os.path.join(TEST_EMODEL_DIR, "emodel-1.hoc")
        assert actual == expected

    def test_get_properties(self):
        actual = self.emodels.get_properties(1)
        expected = {
            "propA": 0.11,
            "propB": 0.12,
        }
        assert actual == expected

    def test_get_mecombo_info(self):
        actual = self.emodels.get_mecombo_info(1)
        expected = {'combo_name': 'mecombo-1', 'morph_name': 'morph-1',
                    'layer': 2, 'fullmtype': 'L23_BP', 'etype': 'bAC',
                    'emodel': 'emodel-1', 'propA': 0.11, 'propB': 0.12}
        assert actual == expected


class TestEModelHelperOldStyle:
    def setup_method(self):
        circuit = Mock()
        circuit.cells.get = _mock_circuit_get
        self.emodels = test_module.EModelHelper(TEST_EMODEL_DIR, circuit)

    def test_get_filepath(self):
        actual = self.emodels.get_filepath(1)
        expected = os.path.join(TEST_EMODEL_DIR, "mecombo-1.hoc")
        assert actual == expected

    def test_get_properties(self):
        actual = self.emodels.get_properties(1)
        assert actual == None

    def test_get_mecombo_info(self):
        actual = self.emodels.get_mecombo_info(1)
        expected = {'combo_name': 'mecombo-1'}
        assert actual == expected
