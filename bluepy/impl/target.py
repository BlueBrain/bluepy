""" Target files parsing. """
import re
import logging
from functools import partial

import numpy as np

from bluepy.exceptions import BluePyError
from bluepy.utils import LazyDict, str2gid, open_utf8

L = logging.getLogger(__name__)

TARGET_REGEX = re.compile(
    r"""
    Target               # Start token
    \s+                  # 1 or more whitespace
    (?P<type>\S+)        # type
    \s+                  # 1 or more whitespace
    (?P<name>\S+)        # name
    \s+                  # 1 or more whitespace
    (?P<start>\{)        # start brace
    (?P<contents>[^}]*)  # not the end brace (contents is needed by brainbuilder)
    (?P<end>})           # end brace
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL
)

SECTION_NAMES = ["soma", "axon"]


class Target:
    """ Target representation. """
    def __init__(self, filename, name, type_, start, end):
        """Initialize a Target object.

        Args:
            filename(str): name of the targets file.
            name (str): name of the target.
            type_ (str): type of the target (Cell, Compartment, Section).
            start (int): starting index of the specific target content in file.
            end (int): ending index of the specific target content in file.
        """
        if type_ not in ('Cell', 'Compartment', 'Section'):
            raise BluePyError(f"Unknown target type: {type_}")

        self.filename = filename
        self.name = name
        self.type = type_
        self.start = start
        self.end = end
        self._children = None  # uninitialized array of children gids

    def is_resolved(self):
        """Return True if the children have been already resolved."""
        return self._children is not None

    def resolve(self, ctx):
        """Resolve the target name and return the sorted array of unique GIDs.

        The array is calculated only the first time, then it's cached and
        the content string is set to None to free the memory.

        Args:
            ctx (TargetContext): target context needed to recursively resolve children.
        """
        if self.is_resolved():
            # return the children array if already resolved
            return self._children
        section_subtarget_count = 0

        with open(self.filename, encoding="utf-8") as fd:
            fd.seek(self.start)
            split_content = fd.read(self.end - self.start).split()

        gids = []  # list of integer gids
        result = []  # list of arrays that will be concatenated together
        for child in split_content:
            try:
                gids.append(str2gid(child))
            except ValueError:
                if child in SECTION_NAMES:
                    section_subtarget_count += 1
                    if len(split_content) < 2:
                        raise BluePyError("Section type need to be defined for a target.")
                    if section_subtarget_count > 1:
                        raise BluePyError("You can only have one Section type for a target.")
                    continue
                result.append(ctx.get_target(child).resolve(ctx=ctx))

        # converting gids to array before calling hstack seems slightly faster,
        # and it ensures that the final array is of type int even when the arrays are empty
        result.append(np.asarray(gids, dtype=int))
        self._children = np.unique(np.hstack(result))
        if len(self._children) == 0:
            L.warning("Target '%s' resolves to empty GID list", self.name)
        else:
            L.debug("Target '%s' resolved to %s GIDs", self.name, len(self._children))
        return self._children


def _parse_target_file(filepath):
    """ Parse .target file, return generator of `Target`s. """
    with open_utf8(filepath) as f:
        contents = f.read()

    for m in TARGET_REGEX.finditer(contents):
        start = m.span('start')[1]
        end = m.span('end')[0]
        yield Target(filepath, m.group('name'), m.group('type'), start, end)


class TargetContext:
    """ Resolving target names to GIDs. """
    def __init__(self, targets):
        self._targets = targets

    @property
    def names(self):
        """ List of available target names. """
        return sorted(self._targets.keys())

    def get_target(self, name):
        """Return the Target with the given name."""
        try:
            return self._targets[name]
        except KeyError:
            raise BluePyError(f"Undefined target: {name}")

    def resolve(self, name):
        """Resolve the target name to sorted array of unique GIDs."""
        return self.get_target(name).resolve(ctx=self)

    @classmethod
    def load(cls, filepaths):
        """ Load a collection of .target files. """
        targets = {}
        for filepath in filepaths:
            for target in _parse_target_file(filepath):
                if target.name in targets:
                    raise BluePyError(f"Target name specified more than once: {target.name}")
                targets[target.name] = target
        return cls(targets)

    def to_node_sets(self, one_based=False):
        """Convert a target file to a node_sets dictionary.

        Args:
            one_based (bool): define if the returned node ids are 1-based or 0-based.

        Notes:
            The 0-based node ids (gids) are used internally in the sonata files and in the full
            sonata circuits when the 1-based gids are used in the nrn and target files and in the
            BlueConfig circuit.
        """
        def _to_node_ids(_key):
            try:
                node_ids = self.resolve(_key)
                if shift:
                    # this will create a copy of the target gids and duplicate the used memory,
                    # because the result is stored in node_sets and the original array is kept
                    node_ids = node_ids + shift
                return {"node_id": node_ids}
            except KeyError:
                L.warning("Cannot convert %s to node sets", _key)
                # It will fail only when trying to access node_id
                return {}

        shift = 0 if one_based else -1
        # Since the node_sets are resolved lazily, node_sets may contain also keys for targets
        # that cannot be converted to node_sets. May it be an issue?
        node_sets = LazyDict({key: partial(_to_node_ids, key) for key in self._targets})
        return node_sets

    def filter(self, names, inplace=False):
        """Only keep the targets defined in names.

        Args:
            names(set): a set of names containing the targets to keep.
            inplace(bool): if True, the operation is done inplace.

        Returns:
            TargetContext/None: the filtered TargetContext if inplace is False, None otherwise.
        """
        filtered_targets = {key: item for key, item in self._targets.items() if key in names}
        if not inplace:
            return TargetContext(filtered_targets)
        self._targets = filtered_targets
        return None
