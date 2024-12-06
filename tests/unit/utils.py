import builtins
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager


@contextmanager
def tmp_file(dirname, content, cleanup=True):
    _, filepath = tempfile.mkstemp(suffix=str(uuid.uuid4()), prefix="BlueConfig_", dir=dirname)
    with open(filepath, "r+") as fd:
        fd.write(content)
    try:
        yield filepath
    finally:
        if cleanup:
            os.remove(filepath)


@contextmanager
def setup_tempdir(cleanup=True):
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)


@contextmanager
def copy_file(filepath):
    """Provides a copy of a file inside a tmp directory

    Returns:
        str: a new file from a tmp dir.
    """
    with setup_tempdir() as tmp_dir:
        output = os.path.join(tmp_dir, os.path.basename(filepath))
        shutil.copy(filepath, output)
        yield output


class PatchImport:
    """Override builtins.__import__ to raise ImportError for the specified modules.

    It has effect only if the modules are imported after calling the setup method.

    Usage inside a test module with pytest:
        patch_import = PatchImport(<module_names>)
        # setup_module and teardown_module are fixtures
        setup_module = patch_import.setup
        teardown_module = patch_import.teardown

    Usage as a context manager:
        with PatchImport(<module_names>):
            <code>
    """

    def __init__(self, *modules):
        """Initialize the module patcher.

        Args:
            *modules: modules to be patched.
        """
        self._modules = set(modules)
        self._real_import = builtins.__import__
        if self._real_import.__name__ != "__import__":
            raise RuntimeError("__import__ cannot be patched multiple times")

    def setup(self):
        def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in self._modules:
                raise ImportError(f"Module {name} is patched and cannot be imported")
            return self._real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _patched_import

    def teardown(self):
        builtins.__import__ = self._real_import

    def __enter__(self):
        self.setup()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.teardown()
