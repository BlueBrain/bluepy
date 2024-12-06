import os.path

import pytest

# import tests that should be run without brion too
from test_compartment_report import (
    TestCompareReaders,
    TestCompareReadersWithCompartmentReport,
    TestTimeSeriesReport,
    TestTimeSeriesReportSonata,
    test_full_fails_report,
)
from utils import PatchImport

import bluepy.impl.compartment_report as test_module
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

# patch the brion module to raise an ImportError
patch_import = PatchImport("brion")
# setup_module and teardown_module are pytest fixtures
setup_module = patch_import.setup
teardown_module = patch_import.teardown


def test_brion_is_not_installed():
    with pytest.raises(ImportError, match="Module brion is patched and cannot be imported"):
        import brion


def test_bbp_reports_without_brion():
    # I cannot convert sonata/h5 to bbp so I consider the minimal testing here for this format
    filepath = os.path.join(TEST_DATA_DIR, "soma.bbp")
    with pytest.raises(BluePyError, match="Brion is mandatory"):
        test_module.TimeSeriesReport(filepath)


# the subclass has the same name as the superclass to trick pytest
class TestCompareReaders(TestCompareReaders):
    def test_impl(self):
        # H5Reader is used instead of BrionReport when brion is not available
        assert isinstance(self._reader_h5._reader, test_module.H5Reader)
        assert isinstance(self._reader_sonata._reader, test_module.LibSonataReader)


# the subclass has the same name as the superclass to trick pytest
class TestCompareReadersWithCompartmentReport(TestCompareReadersWithCompartmentReport):
    def test_impl(self):
        # H5Reader is used instead of BrionReport when brion is not available
        assert isinstance(self._reader_h5._reader, test_module.H5Reader)
        assert isinstance(self._reader_sonata._reader, test_module.LibSonataReader)
