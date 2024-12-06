""" Utilities for BluePy."""
import collections
import itertools
import sys
from collections.abc import Iterable, Mapping
from functools import partial

import numpy as np

# Compliant with PEP 597 and avoid W1514 with pylint (unspecified-encoding)
open_utf8 = partial(open, encoding="utf-8")

# dtypes for the different node and edge ids. We are using np.int64 to avoid the infamous
# https://github.com/numpy/numpy/issues/15084 numpy problem. This type needs to be used for
# all returned node or edge ids.
IDS_DTYPE = np.int64


def str2gid(s):
    """ 'a7' -> 7 """
    if not s or s[0] != "a":  # faster than startswith
        raise ValueError(f"Invalid GID string: '{s}'")
    return int(s[1:])


def gid2str(k):
    """ 7 -> 'a7' """
    return f"a{k}"


def normalize_endianness(a):
    """ Change `a` byte order if it's not same as the native one. """
    native = {'big': '>', 'little': '<'}[sys.byteorder]
    if a.dtype.byteorder in ['=', '|', native]:
        return a
    else:
        return a.byteswap().newbyteorder()


def take_n(it, n):
    """ Take no more than `n` items from `it` iterable. """
    return list(itertools.islice(it, n))


def is_iterable(v):
    """Check if `v` is any iterable (strings are considered scalar)."""
    return isinstance(v, Iterable) and not isinstance(v, str)


def ensure_list(v):
    """ Convert iterable / wrap scalar into list (strings are considered scalar). """
    if is_iterable(v):
        return list(v)
    else:
        return [v]


def group_by_first(pairs):
    """
    Group pairs by first element into multi-map

    Example:
        >>> groupby_first([('a', 0), ('b', 1), ('a', 2)])
        {'a': [0, 2], 'b': [1]}
    """
    result = collections.defaultdict(list)
    for first, second in pairs:
        result[first].append(second)
    return dict(result)


class LazyDict(Mapping):
    """Lazy mapping object where the values are evaluated only when accessed."""

    def __init__(self, *args, **kw):
        """Initialize the object.

        The value given for each key must be a function that will be evaluated
        when the item is accessed.
        """
        self._functions = dict(*args, **kw)
        self._data = {}

    def __getitem__(self, key):
        """Return the requested item, calculating and caching it if needed."""
        try:
            item = self._data[key]
        except KeyError:
            try:
                func = self._functions[key]
            except KeyError:
                raise KeyError(f"KeyError: {key}") from None
            item = self._data[key] = func()
        return item

    def __iter__(self):
        """Iterate over the mapping."""
        return iter(self._functions)

    def __len__(self):
        """Return the length of the mapping."""
        return len(self._functions)
