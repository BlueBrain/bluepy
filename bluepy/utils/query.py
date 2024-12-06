"""Query functions for cells."""
from collections.abc import Mapping
import numpy as np
from bluepy.utils import deprecate
from bluepy.exceptions import BluePyError

NODE_ID_KEY = 'node_id'


def _complex_query(prop, query):
    # pylint: disable=assignment-from-no-return
    result = np.full(len(prop), True)
    for key, value in query.items():
        if key == '$regex':
            result = np.logical_and(result, prop.str.match(value + "\\Z"))
        else:
            raise BluePyError(f"Unknown query modifier: '{key}'")
    return result


def gids_by_filter(cells, props):
    """
    Return index of `cells` rows matching `props` dict.

    `props` values could be:
        pairs (range match for floating dtype fields)
        scalar or iterables (exact or "one of" match for other fields)

    E.g.:
        >>> gids_by_filter(cells, { Cell.X: (0, 1), Cell.MTYPE: 'L1_SLAC' })
        >>> gids_by_filter(cells, { Cell.LAYER: [2, 3] })
    """
    unknown_props = set(props) - (set(cells.columns) | {NODE_ID_KEY})
    if unknown_props:
        raise BluePyError(f"Unknown cell properties: [{', '.join(unknown_props)}]")

    mask = np.full(len(cells), True)
    for prop, values in props.items():
        if prop == NODE_ID_KEY:
            prop = cells.index
        else:
            prop = cells[prop]

        prop_mask = False
        if issubclass(prop.dtype.type, np.floating):
            v1, v2 = values
            prop_mask = np.logical_and(prop >= v1, prop <= v2)
        elif isinstance(values, str) and values.startswith('@'):
            deprecate.fail("""
            '@<value>' syntax for regular expressions is removed;
            Please use {'$regex': <value>} instead.
            """)
        elif isinstance(values, str) and values.startswith('regex:'):
            deprecate.fail("""
            'regex:<value>' syntax for regular expressions is removed;
            Please use {'$regex': <value>} instead.
            """)
        elif isinstance(values, Mapping):
            prop_mask = _complex_query(prop, values)
        else:
            prop_mask = np.in1d(prop, values)
        mask = np.logical_and(mask, prop_mask)

    return cells.index[mask].values
