""" Morphology-related classes/methods. """
import os

import morph_tool.transform as transformations
import morphio
import numpy as np
import pandas as pd

from cached_property import cached_property

from bluepy.exceptions import require, BluePyError
from bluepy.utils.url import get_file_path_by_url
from bluepy.enums import Cell, Section, Segment


def segment_volume(p0, p1, r0, r1):
    """Compute the volume of a segment."""
    h = np.linalg.norm(p0 - p1)
    return np.pi * h * ((r0 * r0) + (r0 * r1) + (r1 * r1)) / 3.0


def segment_volumes(section):
    """Compute all the volume of a section's segments."""
    radius = section.diameters * 0.5
    points = section.points
    segments = zip(points[:-1], points[1:], radius[:-1], radius[1:])
    return np.array([segment_volume(p0, p1, r0, r1) for p0, p1, r0, r1 in segments])


def section_volume(section):
    """Volume of a section."""
    return sum(segment_volumes(section))


def segment_midpoints(section):
    """ Find segment midpoints. """
    return 0.5 * (section.points[:-1] + section.points[1:])


def segment_lengths(section):
    """Find segment lengths."""
    return np.linalg.norm(np.diff(section.points, axis=0), axis=1)


def section_length(section):
    """Find section lengths."""
    return segment_lengths(section).sum()


def section_branch_order(section):
    """Find section branch order."""
    return sum(1 for _ in section.iter(morphio.IterType.upstream)) - 1


def section_path_length(section):
    """Find section path length."""
    return sum(section_length(prev_section)
               for prev_section in section.iter(morphio.IterType.upstream))


def _dispatch():
    """Dispatch function for the morphology formats.

    Notes:
        the dict contains both the historical bluepy format keys (h5v1, ascii, swc) and add the
        ones from the blueconfig specs (h5, asc, swc). The dict values are the directory lookups.
    """
    dispatch = {
        'h5v1': (['', 'h5v1', 'v1'], 'h5'),
        'ascii': (['', 'ascii'], 'asc'),
        'swc': (['', 'swc'], 'swc'),
    }
    dispatch['h5'] = dispatch['h5v1']
    dispatch['asc'] = dispatch['ascii']
    return dispatch


class MorphHelper:
    """ Collection of morphology-related methods. """

    def __init__(self, url, circuit, spatial_index=None, morph_type="h5v1"):
        self._morph_path = get_file_path_by_url(url)
        self._morph_type = morph_type
        self._dispatch = _dispatch()
        self._circuit = circuit
        self._index_url = spatial_index

    def get_filepath(self, gid, source=None):
        """ Return path to H5 morphology file corresponding to `gid`. """
        morph_format = source if source else self._morph_type
        try:
            dirnames, file_ext = self._dispatch[morph_format]
        except KeyError:
            raise BluePyError(
                f"'{morph_format}' is not a valid morphology format. Please use one of: "
                f"{tuple(self._dispatch.keys())}"
            )
        name = self._circuit.cells.get(gid, Cell.MORPHOLOGY)
        filename = f"{name}.{file_ext}"
        for dirname in dirnames:
            filepath = os.path.join(self._morph_path, dirname, filename)
            if os.path.exists(filepath):
                return filepath
        raise BluePyError(f"Can not locate '{filename}' in '{self._morph_path}'")

    def get(self, gid, transform=False, source=None):
        """
        Return MorphIO morphology object corresponding to `gid`.

        If `transform` is True, rotate and translate morphology points
        according to `gid` position in the circuit.

        Note: a `MorphIO` object is returned.  That library does not consider the
              soma a `section`, in the sense that the simulator does.  ie:
                * the soma is not returned while iterating over the object
                * the `section_ids` of the sections read from the file after
                  the soma need to be decremented by 1 because the soma is skipped
              Thus, the `section_id`s returned from the connectivity must be _aligned_
              if the `MorphIO` section access functions are used.  If a
              `section_id` is zero, the soma should be examined.  For all other
              `section_id`s, their value must be decreased by one so that they
              match the `MorphIO` convention.  This all assumes that the soma is
              the first `section` in the file, and contains only one section, which
              *should* be the case for morphologies within BBP.
        """
        filepath = self.get_filepath(gid, source=source)
        if transform:
            result = morphio.mut.Morphology(filepath, options=morphio.Option.nrn_order)
            T = np.eye(4)
            T[:3, :3] = self._circuit.cells.orientations(gid)  # rotations
            T[:3, 3] = self._circuit.cells.positions(gid).values  # translations
            transformations.transform(result, T)
            result = result.as_immutable()
        else:
            result = morphio.Morphology(filepath, options=morphio.Option.nrn_order)

        return result

    def section_features(self, gid, features):
        """
        Get section features for the given `gid`.

        Returns pandas DataFrame indexed by section_id.
        """
        morph = self.get(gid, transform=False)
        sections = morph.sections
        result = pd.DataFrame(index=pd.Index([s.id for s in sections], name=Section.ID))

        if Section.NEURITE_TYPE in features:
            result[Section.NEURITE_TYPE] = [s.type for s in sections]

        if Section.BRANCH_ORDER in features:
            result[Section.BRANCH_ORDER] = [section_branch_order(s) for s in sections]

        if Section.LENGTH in features:
            result[Section.LENGTH] = [section_length(s) for s in sections]

        if Section.NEURITE_START_DISTANCE in features:
            result[Section.NEURITE_START_DISTANCE] = [
                (section_path_length(s) - section_length(s)) for s in sections
            ]

        # MorphIO doesn't consider the soma a section; makes sure we match the
        # section ids from the edge file (whether nrn.h5 or SONATA)
        result.index += 1

        if morph.soma:
            soma = pd.DataFrame(data=0,
                                index=pd.Index([0], name=Section.ID),
                                columns=result.columns)
            if Section.NEURITE_TYPE in features:
                soma[Section.NEURITE_TYPE] = morphio.SectionType.soma
            result = pd.concat([soma, result])

        return result

    def segment_features(self, gid, features, atlas=None):
        """
        Get segment features for the given `gid`.

        Returns pandas DataFrame multi-indexed by (section_id, segment_id)
        """
        morph = self.get(gid, transform=Segment.REGION in features)

        chunks = {}
        for section in morph.sections:
            values = {}

            if Segment.LENGTH in features or Segment.SECTION_START_DISTANCE in features:
                lengths = segment_lengths(section)
                if Segment.LENGTH in features:
                    values[Segment.LENGTH] = lengths
                if Segment.SECTION_START_DISTANCE in features:
                    values[Segment.SECTION_START_DISTANCE] = np.cumsum(lengths) - lengths

            if Segment.REGION in features:
                require(atlas is not None, "Atlas is required for calculating segment regions")
                values[Segment.REGION] = atlas.lookup(segment_midpoints(section), outer_value=-1)

            # MorphIO doesn't consider the soma a section; makes sure we match the
            # section ids from the edge file (whether nrn.h5 or SONATA)
            chunks[section.id + 1] = values

        assert 0 not in chunks, "Should not have already have a zero section"
        if morph.soma:
            chunks[0] = {feature: np.float32(0) for feature in features}

            if Segment.REGION in features:
                require(atlas is not None, "Atlas is required for calculating segment regions")
                chunks[0][Segment.REGION] = atlas.lookup(morph.soma.center[np.newaxis],
                                                         outer_value=-1)

        result = pd.DataFrame()
        for feature in features:
            result[feature] = pd.concat({
                section_id: pd.Series(values[feature])
                for section_id, values in chunks.items()
            })

        result.index.rename([Section.ID, Segment.ID], inplace=True)

        return result

    def segment_points(self, gid, neurite_type=None, transform=False):
        """
        Get segment points for given `gid`.

        Args:
            gid: GID of interest
            neurite_type (morphio.SectionType): neurite type of interest
            transform (bool): rotate / translate according to GID position in space

        Returns:
            pandas DataFrame multi-indexed by (Section.ID, Segment.ID);
            and Segment.[X|Y|Z][1|2] as columns.

        Note: soma is returned as a spherical segment
        """
        morph = self.get(gid, transform=transform)

        index = []
        chunks = []
        if neurite_type is None:
            it = morph.iter()
        else:
            it = (x for x in morph.iter() if x.type == neurite_type)

        for sec in it:
            pts = sec.points
            chunk = np.zeros((len(pts) - 1, 6))
            chunk[:, 0:3] = pts[:-1]
            chunk[:, 3:6] = pts[1:]
            chunks.append(chunk)

            # MorphIO doesn't consider the soma a section; makes sure we match the
            # section ids from the edge file (whether nrn.h5 or SONATA)
            index.extend((sec.id + 1, seg_id) for seg_id in range(len(pts) - 1))

        # MorphIO doesn't consider the soma as as segment, manually make a spherical one
        if morph.soma and neurite_type in (None,
                                           morphio.SectionType.all,
                                           morphio.SectionType.soma,
                                           ):
            index.append((0, 0))
            chunk = np.concatenate((morph.soma.center, morph.soma.center))[np.newaxis]
            chunks.append(chunk)

        if index:
            result = pd.DataFrame(
                data=np.concatenate(chunks),
                index=pd.MultiIndex.from_tuples(index, names=[Section.ID, Segment.ID]),
                columns=[
                    Segment.X1, Segment.Y1, Segment.Z1,
                    Segment.X2, Segment.Y2, Segment.Z2,
                ]
            )
        else:
            # no sections with specified neurite type
            result = pd.DataFrame()

        return result

    @cached_property
    def spatial_index(self):
        """ Spatial index. """
        require(self._index_url is not None, "Spatial index not defined")
        from bluepy.index import SegmentIndex
        return SegmentIndex(os.path.join(get_file_path_by_url(self._index_url), "SEGMENT"))
