import os
from unittest.mock import patch

import pytest

import bluepy.settings as test_module
from bluepy.utils import deprecate


def test_str2bool():
    assert test_module.str2bool('1')
    assert test_module.str2bool('y')
    assert test_module.str2bool('YES')
    assert not test_module.str2bool('0')
    assert not test_module.str2bool('n')
    assert not test_module.str2bool('NO')
    assert not test_module.str2bool(None)


@patch("bluepy.settings.STRICT_MODE", None)
def test_STRICT_MODE():
    with patch.dict(os.environ, {'BLUEPY_STRICT_MODE': '1'}):
        test_module.load_env()
    assert test_module.STRICT_MODE == True
    pytest.raises(deprecate.BluePyDeprecationError, deprecate.warn)


@patch("bluepy.settings.ATLAS_CACHE_DIR", None)
def test_ATLAS_CACHE_DIR():
    with patch.dict(os.environ, {'BLUEPY_ATLAS_CACHE_DIR': 'foo'}):
        test_module.load_env()
    assert test_module.ATLAS_CACHE_DIR == 'foo'

