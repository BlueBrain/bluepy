import itertools
from functools import partial
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pytest

from bluepy import utils


def test_gid2str():
    assert utils.gid2str(42) == "a42"


def test_str2gid():
    assert utils.str2gid("a42") == 42
    pytest.raises(ValueError, utils.str2gid, "a42z")
    pytest.raises(ValueError, utils.str2gid, "L23")
    pytest.raises(ValueError, utils.str2gid, "23")
    pytest.raises(ValueError, utils.str2gid, "")


def test_normalize_endianness_1():
    a = np.array([1, 2], dtype='=f4')
    with patch('sys.byteorder', 'big'):
        b = utils.normalize_endianness(a)
        assert a is b


def test_normalize_endianness_2():
    a = np.array([1, 2], dtype='>f4')
    with patch('sys.byteorder', 'big'):
        b = utils.normalize_endianness(a)
        assert a is b


def test_normalize_endianness_3():
    a = np.array([1, 2], dtype='>f4')
    with patch('sys.byteorder', 'little'):
        b = utils.normalize_endianness(a)
        assert b.dtype == '<f4'
        npt.assert_equal(a, b)


def test_take_n():
    assert utils.take_n(itertools.count(), 3) == [0, 1, 2]


def test_ensure_list():
    assert utils.ensure_list(1) == [1]
    assert utils.ensure_list([1]) == [1]
    assert utils.ensure_list((2, 1)) == [2, 1]
    assert utils.ensure_list('abc') == ['abc']


def test_group_by_first():
    assert utils.group_by_first([]) == {}
    assert utils.group_by_first([('a', 0), ('b', 1), ('a', 2)]) == {'a': [0, 2], 'b': [1]}


def test_lazydict():
    def _func1():
        return 2 ** 5

    def _func2(exp):
        return 2 ** exp

    _func3 = Mock(return_value=999)

    ld = utils.LazyDict(
        {
            "a": lambda: 2 ** 4,
            "b": _func1,
            "c": partial(_func2, exp=6),
            "d": _func3,
        }
    )

    assert len(ld) == 4
    assert list(ld) == ["a", "b", "c", "d"]
    assert _func3.called == 0

    assert ld["a"] == 16
    assert ld["b"] == 32
    assert ld["c"] == 64
    assert ld["d"] == 999

    assert _func3.called == 1
    # verify that the value is cached and the function is not called again
    assert ld["d"] == 999
    assert _func3.called == 1

    with pytest.raises(KeyError, match="KeyError: 123"):
        ld[123]
