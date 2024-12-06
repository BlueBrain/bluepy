""" Utilities for interacting with the spatial indexer."""

import os
import itertools as it

import numpy
from bluepy.exceptions import BluePyError

try:
    import libFLATIndex as FI
except ImportError as e:
    raise BluePyError("Please install libflatindex to use the indexing feature. "
                      "pip install -i https://bbpteam.epfl.ch/repository/devpi/simple libflatindex")


return_val_docs = """
        The query returns columns as follows, depending on the index type

        For mesh info:

        [vertex1.X,
         vertex1.Y,
         vertex1.Z,
         vertex2.X,
         vertex2.Y,
         vertex2.Z,
         vertex3.X,
         vertex3.Y,
         vertex3.Z,
         neuronID,
         vertex1-bbpIndex,
         vertex2-bbpIndex,
         vertex3-bbpIndex]

        For segment info:

        [Cone-begin.X,
         Cone-begin.Y,
         Cone-begin.Z,
         Cone-end.X,
         Cone-end.Y,
         Cone-end.Z,
         Cone-beginRadius,
         Cone-endRadius,
         neuronID,
         sectionID,
         segmentID
         BBP-SDK_sectionTypeID]

        For Synapse info:

         [PreSynPos.x (not implemented, is == PostSynPos.x)
          PreSynPos.y (not implemented, is == PostSynPos.y)
          PreSynPos.z (not implemented, is == PostSynPos.z)
          PostSynPos.x
          PostSynPos.y
          PostSynPos.z
          spineLength (not implemented, =1um)
          BBP-SDK_synapseID,
          SynCounter (Synapse Counter number.\
          Numbering done while reading the data, starting from zero)
          PreSynNeuronGID
          PostSynNeuronGID
          IsExcitatory]

       For Soma info:

         [CenterPos.x
          CenterPos.y
          CenterPos.z
          Radius
          NeuronGID]

"""


class DIASIndex:
    """ DIASIndex class """
    postfixes = ['_index.dat', '_index.idx', '_payload.dat']

    def __init__(self, index_prefix):
        """index_prefix is prefix to index files
        e.g. '/scratch/SpatialIndexing/DIAS/M'

        appends ['_index.dat', '_index.idx', '_payload.dat']
        to it to load index

        """
        self.prefix = index_prefix

        # if index exists, load it

        if self.exists():
            self.load()
        else:
            self.index = None

    @classmethod
    def _wrap_result(cls, result):
        """ Wrap query response hook. """
        return result

    def load(self):
        """loads index given by self.prefix"""
        self.index = FI.loadIndex(str(self.prefix))  # pylint: disable=no-member
        if not self.index:
            raise IOError("DIASIndex.load_index: Could not load index.")

    def exists(self):
        """checks that all files exist for index given prefix."""
        for postfix in self.postfixes:
            if not os.path.exists(self.prefix + postfix):
                return False
        return True

    def unload(self):
        """unloads the index."""
        self.check_valid_loaded()
        # Disabled until SPIND-64 is fixed
        # FI.unLoadIndex(self.index)
        self.index = None

    def check_valid_loaded(self):
        """checks if a valid index is loaded."""
        if not self.index:
            raise RuntimeError("DIASIndex.check_valid_index: no valid index loaded yet.")

    def q_vicinity(self, center_tuple, radius):
        """
        Vicinity Query

        center = (x, y, z): Coordinates where to start search
        can also be a numpy array of len==3

        radius: Maximum extent of finding results
        returns :list of mesh information / list of segment information

        """
        self.check_valid_loaded()
        assert len(center_tuple) == 3

        x, y, z = float(center_tuple[0]), float(center_tuple[1]), float(center_tuple[2])

        q = FI.vicinityQuery(self.index, x, y, z, radius)  # pylint: disable=no-member
        res = numpy.fromiter(it.chain.from_iterable(q), dtype=numpy.float).reshape((len(q), -1))
        return self._wrap_result(res)

    q_vicinity.__doc__ += return_val_docs  # pylint: disable=E1101

    def q_window(self, low_tuple, high_tuple):
        """
        Window Query
        Objects are returned if the object bounding box is inside the query box.
        See also: q_window_oncenter

        low = (x, y, z): Minimum Coordinates of an Axis Aligned Box
        can also be a numpy array of len==3

        high = (x, y, z): Maximum coordinates of an Axis Alligned Box
        can also be a numpy array of len==3

        returns: list of mesh information / list of segment information

        """

        self.check_valid_loaded()
        assert len(low_tuple) == 3
        # support numpy array
        if isinstance(low_tuple, numpy.ndarray):
            low_tuple = tuple(map(float, low_tuple))
        assert len(high_tuple) == 3
        # support numpy array
        if isinstance(high_tuple, numpy.ndarray):
            high_tuple = tuple(map(float, high_tuple))
        arg_tuple = low_tuple + high_tuple
        try:
            result = FI.numpy_windowQuery(self.index, *arg_tuple)  # pylint: disable=no-member
        except IndexError:
            # temporary workaround for SPIND-62
            result = []
        return self._wrap_result(result)

    q_window.__doc__ += return_val_docs  # pylint: disable=E1101

    def q_window_oncenter(self, low_tuple, high_tuple):
        """
        Window Query
        "On Center" version (objects are returned only if their mid point is inside the query box)
        See also: q_window

        low = (x, y, z): Minimum Coordinates of an Axis Aligned Box
        can also be a numpy array of len==3

        high = (x, y, z): Maximum coordinates of an Axis Alligned Box
        can also be a numpy array of len==3

        returns: list of mesh information / list of segment information

        """

        self.check_valid_loaded()
        assert len(low_tuple) == 3
        # support numpy array
        if isinstance(low_tuple, numpy.ndarray):
            low_tuple = tuple(map(float, low_tuple))
        assert len(high_tuple) == 3
        # support numpy array
        if isinstance(high_tuple, numpy.ndarray):
            high_tuple = tuple(map(float, high_tuple))
        arg_tuple = low_tuple + high_tuple
        try:
            # pylint: disable=no-member
            result = FI.numpy_windowQueryOnCenter(self.index, *arg_tuple)
        except IndexError:
            # temporary workaround for SPIND-62
            result = []
        return self._wrap_result(result)

    q_window_oncenter.__doc__ += return_val_docs  # pylint: disable=E1101

    def __del__(self):
        # unload index if valid loaded
        if self.index is not None:
            self.unload()
