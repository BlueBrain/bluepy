import logging
import os
from pathlib import Path

import pytest
from notebook_tester import NotebookTester

L = logging.getLogger(__name__)

TESTING_LOG_LEVEL = os.getenv("TESTING_LOG_LEVEL", "INFO")
TESTING_MATCH_FILES = os.getenv("TESTING_MATCH_FILES", "*")
TESTING_FORCE_UPDATE = int(os.getenv("TESTING_FORCE_UPDATE", "0"))
NOTEBOOKS_PATH = Path(__file__).parent.joinpath("../../examples/ipython_notebooks").resolve()


# convert paths to strings to nicely print them during the test execution
notebooks = [
    str(path) for path in sorted(NOTEBOOKS_PATH.glob("*.ipynb")) if path.match(TESTING_MATCH_FILES)
]


@pytest.mark.skipif(not TESTING_FORCE_UPDATE, reason="Not overwriting files")
@pytest.mark.parametrize("path", notebooks)
def test_update_notebook(path, caplog):
    """Update the notebooks with the generated output.

    It should be executed only to update the notebooks when a test is added or modified,
    or the output changes for a known reason (for example because a library was updated).
    """
    caplog.set_level(TESTING_LOG_LEVEL)
    NotebookTester(path).update_notebook()


@pytest.mark.parametrize("path", notebooks)
def test_check_notebook(path, caplog):
    """Run the notebooks and check the generated output.

    These tests take and run "bluepy/examples/ipython_notebooks/" examples.
    Results from the tests are compared with the original notebook results.
    It only compares png and text/html outputs.
    """
    caplog.set_level(TESTING_LOG_LEVEL)
    NotebookTester(path).check_notebook()
