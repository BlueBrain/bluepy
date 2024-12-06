"""Execute and check an Ipython Notebook file."""
import base64
import io
import logging
import os
import re
import time
from copy import deepcopy
from difflib import Differ
from itertools import zip_longest
from pathlib import Path

import imageio
import nbclient
import nbformat
import numpy as np

L = logging.getLogger(__name__)
PNG_SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "png_diff")


def _fuzzy_compare(value1, value2):
    """Fuzzy compares two values (apply sanitization if values are strings).

    If values are strings, they are sanitized to avoid problems with random values
    like hex addresses or uids. Direct compare if values are something else than strings.

    Inputs:
        value1 : first value to compare
        value2 : second value to compare

    Returns:
        boolean value for the comparison
    """

    def _sanitize(s):
        """Sanitizes string for comparison."""
        # ignore trailing newlines
        s = s.rstrip("\r\n")
        # normalize hex addresses:
        s = re.sub(r"0x[A-Fa-f0-9]+", "0xFFFFFFFF", s)
        # normalize UUIDs:
        s = re.sub(r"[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}", "U-U-I-D", s)
        # for warnings
        s = re.sub(r"/gpfs/.*Warning:.*", "WARNING", s)
        return s

    if not type(value1) is type(value2):
        return False
    if isinstance(value1, str):
        return _sanitize(value1) == _sanitize(value2)
    return value1 == value2


def _is_diff_png(b64_ref, b64_test, save=True, output_dir=None):
    """Compares the pixels of two PNGs using numpy.

    Computes the difference pixel-wise between two images and saves it in output_dir if needed.

    Args:
        b64_ref: base 64 png string for the reference image.
        b64_test: base 64 png string for the tested image.
        save: boolean value to save the files and the differences between them.
        output_dir: directory to store the reference/tested file and difference.
            If save=True and output_dir=None then the cwd is used.

    Returns:
        True if images are different, False otherwise
    """

    def _png_b64_to_ndarray(b64_png):
        """convert PNG output into a np.ndarray using imageio"""
        decoded_string = base64.decodebytes(b64_png.encode("utf-8"))
        s_io = io.BytesIO(decoded_string)
        return imageio.get_reader(s_io).get_data(0)

    reference, tested = map(_png_b64_to_ndarray, (b64_ref, b64_test))
    if reference.shape != tested.shape:
        if save:
            output_dir = "." if output_dir is None else output_dir
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "reference.png"), "wb") as fd:
                fd.write(base64.decodebytes(b64_ref.encode("utf-8")))
            with open(os.path.join(output_dir, "tested.png"), "wb") as fd:
                fd.write(base64.decodebytes(b64_test.encode("utf-8")))

        return True

    difference = np.abs(reference - tested)
    res = np.count_nonzero(difference) > 0

    if res and save:
        output_dir = "." if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, f"ipynb_tester-{b64_ref[:4]}")

        imageio.imwrite(prefix + "_reference.png", reference)
        imageio.imwrite(prefix + "_tested.png", tested)
        imageio.imwrite(prefix + "_difference.png", 255 - difference)

    return res


def _compare_outputs(
    reference_output,
    tested_output,
    cell_id,
    skip_compare=("metadata", "traceback", "latex", "prompt_number"),
):
    """Compares the output of two different runs of the same cell.

    Notes:
        Only address text, numbers and png images, other are skipped. If a problem is found
        then an Exception is raised.

    Inputs:
        reference_output : the reference dictionary
        tested_output : the tested dictionary
        cell : the current cell from notebook

    Raises:
        if a difference has been found
    """
    for key in reference_output:
        reference = reference_output[key]
        tested = tested_output.get(key)
        if key not in tested_output:
            raise NotebookTesterError(
                f"Execution id {cell_id}, the '{key}' key is present in ref's "
                f"cell output but not in test cell"
            )
        elif key == "data":
            _compare_outputs(reference, tested, cell_id)
        elif key == "image/png":
            if _is_diff_png(reference, tested, save=True, output_dir=PNG_SAVE_DIRECTORY):
                raise NotebookTesterError(
                    f"Execution id {cell_id}, png images differ. "
                    f"Images diff are stored in: {PNG_SAVE_DIRECTORY}"
                )
        elif key not in skip_compare:
            if _fuzzy_compare(reference, tested):
                continue
            if isinstance(reference, str) and isinstance(tested, str):
                diff = Differ().compare(reference.splitlines(True), tested.splitlines(True))
                msg = f"Execution id {cell_id}, text outputs differ:\n" + "".join(diff)
                raise NotebookTesterError(msg)
            raise NotebookTesterError(f"Execution id {cell_id}, {reference} != {tested}")


class NotebookTesterError(Exception):
    """Generic NotebookTester exception."""


class NotebookTester:
    """Execute and check an Ipython Notebook file."""

    def __init__(
        self,
        nb_path,
        timeout=600,
        kernel_name="python3",
        notebook_version=nbformat.NO_CONVERT,
        resources=None,
    ) -> None:
        """Initialize the NotebookTester.

        Args:
            nb_path (str or Path): path to the tested notebook.
            timeout (float): execution timeout.
            kernel_name (str): name of the kernel.
            notebook_version: target version of the notebook.
            resources: dict of resources, for example: {"metadata": {"path": "notebooks/"}}
        """
        self._nb_path = Path(nb_path)
        self._timeout = timeout
        self._kernel_name = kernel_name
        self._notebook_version = notebook_version
        self._resources = resources or {}
        self._start_times = {}

    def _load_notebook(self) -> nbformat.NotebookNode:
        return nbformat.read(self._nb_path, as_version=self._notebook_version)

    def _run_notebook(self, nb: nbformat.NotebookNode) -> None:
        client = nbclient.NotebookClient(
            nb,
            timeout=self._timeout,
            kernel_name=self._kernel_name,
            resources=self._resources,
            record_timing=False,
            coalesce_streams=True,
            on_cell_start=self._on_cell_start,
            on_cell_executed=self._on_cell_executed,
        )
        client.execute()

    def _save_notebook(self, nb: nbformat.NotebookNode) -> None:
        nbformat.write(nb, self._nb_path)

    def _on_cell_start(self, cell, cell_index):
        self._start_times[cell_index] = time.monotonic()
        # remove any execution timing if recorded in a previous run
        cell.get("metadata", {}).pop("execution", None)
        L.info("Executing cell %s...", cell_index)

    def _on_cell_executed(self, cell, cell_index, execute_reply):
        exec_time = time.monotonic() - self._start_times[cell_index]
        L.info("Executed cell %s in %.3f seconds", cell_index, exec_time)

    @staticmethod
    def _build_outputs(nb):
        """Return a dict of outputs from the given notebook."""
        return {
            cell["execution_count"]: deepcopy(cell.outputs)
            for cell in nb.cells
            if cell.cell_type == "code" and cell["execution_count"] is not None
        }

    def check_notebook(self):
        """Run the notebook and check the output.

        It runs all the code cells in the notebook and checks if any error occurs.
        If no code error is raised, then the output produced by each cell (text and png)
        is compared with the output that was previously stored in the cell.
        If a disagreement between the two is found, an exception is raised.

        Important: cells containing the tag skip-execution are skipped.
        """
        nb = self._load_notebook()
        references = self._build_outputs(nb)
        self._run_notebook(nb)
        outputs = self._build_outputs(nb)

        cell_ids = sorted(set(references).union(outputs))
        for cell_id in cell_ids:
            if cell_id not in references:
                raise NotebookTesterError(f"Cell id {cell_id} missing in references")
            if cell_id not in outputs:
                raise NotebookTesterError(f"Cell id {cell_id} missing in outputs")
            for ref, out in zip_longest(references[cell_id], outputs[cell_id], fillvalue={}):
                _compare_outputs(ref, out, cell_id)

    def update_notebook(self):
        """Run the notebook and overwrite the file with the updated output.

        It runs all the code cells in the notebook and checks if any error occurs.
        If no code error is raised, then the notebook is overwritten.

        Important: cells containing the tag skip-execution are skipped.
        """
        nb = self._load_notebook()
        self._run_notebook(nb)
        self._save_notebook(nb)
