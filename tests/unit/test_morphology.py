import os
import sys
from unittest.mock import Mock, patch

import morphio
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from utils import setup_tempdir

import bluepy.morphology as test_module
from bluepy.circuit import Circuit
from bluepy.enums import Section, Segment
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

TEST_MORPH = "small_morph"


class TestMorphHelper:
    def setup_method(self):
        circuit = Mock()
        circuit.cells.get = lambda _, prop: TEST_MORPH
        circuit.cells.positions = lambda _: pd.Series({
            'x': 100.0,
            'y': 200.0,
            'z': 300.0,
        })
        circuit.cells.orientations = lambda _: -np.identity(3)

        self.morph_path = os.path.join(TEST_DATA_DIR, "morphs")
        self.morph = test_module.MorphHelper(self.morph_path, circuit, spatial_index='stub')

        circuit2 = Mock()
        circuit2.cells.get = lambda _, prop: TEST_MORPH
        circuit2.cells.positions = lambda _: pd.Series({
            'x': 100.0,
            'y': 200.0,
            'z': 300.0,
        })
        circuit2.cells.orientations = lambda _: np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.morph2 = test_module.MorphHelper(self.morph_path, circuit2, spatial_index='stub')

    def test_get_filepath(self):
        actual = self.morph.get_filepath(1)
        expected = os.path.join(self.morph_path, "v1", TEST_MORPH + ".h5")
        assert actual == expected

    @patch('os.path.exists', side_effect=[False, True])
    def test_get_filepath_ascii(self, _):
        actual = self.morph.get_filepath(1, source='ascii')
        expected = os.path.join(self.morph_path, "ascii", TEST_MORPH + ".asc")
        assert actual == expected

    def test_get_filepath_raises(self):
        # cannot find the data
        pytest.raises(BluePyError, self.morph.get_filepath, 1, source='swc')
        # invalid format
        pytest.raises(BluePyError, self.morph.get_filepath, 1, source='unknown')

    def test_get(self):
        actual = self.morph.get(1).points
        assert len(actual) == 14

        #  the soma points are excluded. That is, for the test example the 4th first points from the morph are discarded
        expected = [
            [0., 5., 0.],
            [2., 9., 0.],
        ]
        npt.assert_almost_equal(actual[:2], expected)

    def test_get_with_source(self):
        h5 = self.morph.get(1).points
        # this is the same file converted and curated so some of the sections have been merged
        # the first points are identical
        asc = self.morph.get(1, source="ascii").points
        assert h5.shape[0] == 14
        assert asc.shape[0] == 13
        npt.assert_almost_equal(h5[:3], asc[:3])

    def test_get_transform(self):
        actual = self.morph.get(1, transform=True).points
        assert len(actual) == 14
        expected = [
            [100., 195., 300.],
            [ 98., 191., 300.],
        ]
        npt.assert_almost_equal(actual[:2], expected)

        # rotation around the x axis 90 degrees counter clockwise (swap Y and Z)
        # [  0.   5.   0.]
        # [  2.   9.   0.]
        # x = X + 100, y = Z + 200, z = Y + 300, radius does not change
        actual = self.morph2.get(1, transform=True).points
        assert len(actual) == 14
        expected = [
            [100., 200., 305.],
            [102., 200., 309.],
        ]
        npt.assert_almost_equal(actual[:2], expected)

    def test_section_features(self):
        features = [
            Section.NEURITE_TYPE,
            Section.BRANCH_ORDER,
            Section.LENGTH,
            Section.NEURITE_START_DISTANCE,
        ]
        actual = self.morph.section_features(1, features)
        assert len(actual) == 6
        expected = pd.DataFrame(
            data=[
                [morphio.SectionType.soma, 0, 0.0, 0.0],
                [morphio.SectionType.axon, 0, 8.944271, 0.0],
                [morphio.SectionType.axon, 1, 4.0, 8.944271],
                [morphio.SectionType.basal_dendrite, 0, 6.0, 0.0],  # 3, 0, dist(-4, -10), from soma
            ],
            columns=features,
            index=pd.Index([0, 1, 2, 3], name=Section.ID)
        )
        pdt.assert_frame_equal(actual.iloc[:4], expected, check_dtype=False)

    def test_segment_features(self):
        atlas = Mock()
        def lookup(p, outer_value):
            return p[:, 1]
        atlas.lookup = lookup

        features = [Segment.LENGTH, Segment.SECTION_START_DISTANCE, Segment.REGION]
        actual = self.morph.segment_features(1, features, atlas=atlas)
        assert len(actual) == 10
        expected = pd.DataFrame(
            data=[
                [4.472136, 0.0,      193.0],   #  dist([0, 5, 0],[2, 9, 0]), from soma
                [4.472136, 4.472136, 189.0],   #  dist([2, 9, 0], [0, 13, 0]), dist([0, 5, 0],[2, 9, 0])
                [2.0,      0.0,      187.0],   #  dist([0, 13, 0], [2, 13, 0]), from soma
            ],
            columns=features,
            index=pd.MultiIndex.from_tuples([(1, 0), (1, 1), (2, 0)],
                                            names=[Section.ID, Segment.ID,])
        ).astype(np.float32)
        pdt.assert_frame_equal(actual.iloc[:3], expected)
        npt.assert_equal([0., 0., 200.], actual.loc[0, 0].values)

    def test_segment_points_1(self):
        actual = self.morph.segment_points(1).sort_index().head(4)
        expected = pd.DataFrame(
            data=[
                [0., 0., 0., 0., 0., 0.],   # soma
                [0., 5., 0., 2., 9., 0.],
                [2., 9., 0., 0., 13., 0.],
                [0., 13., 0., 2., 13., 0.],
            ],
            columns=[
                Segment.X1, Segment.Y1, Segment.Z1,
                Segment.X2, Segment.Y2, Segment.Z2,
            ],
            index=pd.MultiIndex.from_tuples([(0, 0), (1, 0), (1, 1), (2, 0)],
                                            names=[Section.ID, Segment.ID])
        )
        pdt.assert_frame_equal(actual, expected)

    def test_segment_points_2(self):
        actual = self.morph.segment_points(1, neurite_type=morphio.SectionType.basal_dendrite).head(3)
        expected = [
            [3., -4., 0., 3., -6., 0.],
            [3., -6., 0., 3., -8., 0.],
            [3., -8., 0., 3., -10., 0.],
        ]
        npt.assert_equal(expected, actual.values)

    def test_segment_points_3(self):
        actual = self.morph.segment_points(1, neurite_type=morphio.SectionType.apical_dendrite).head(3)
        assert actual.empty

    def test_segment_points_4(self):
        actual = self.morph.segment_points(1, transform=True)
        expected = [
            [100., 195., 300., 98., 191., 300.],
            [98., 191., 300., 100., 187., 300.],
            [100., 187., 300., 98., 187., 300.],
        ]
        npt.assert_equal(expected, actual.head(3).values)

        npt.assert_equal([100., 200., 300., 100., 200., 300.],
                         actual.loc[0, 0].values)

    @patch.dict(sys.modules, {'libFLATIndex': 'foo'})
    def test_spatial_index(self):
        assert not self.morph.spatial_index.exists()

    @patch('os.path.exists', return_value=True)
    def test_morph_type(self, mock):
        def _test_morphs(morph_type, expected_morph):
            with setup_tempdir() as tmp:
                simple_blueconfig = """
Run Default
{{
    CircuitPath {}
    nrnPath {}
    CellLibraryFile circuit.mvd3
    METypePath {}
    MorphologyPath {}
    {} {}
}}
                                """.format(TEST_DATA_DIR,
                                           TEST_DATA_DIR,
                                           TEST_DATA_DIR,
                                           TEST_DATA_DIR,
                                           "MorphologyType" if morph_type else '',
                                           morph_type if morph_type else '')
                config_path = os.path.join(tmp, 'CircuitConfig')
                with open(config_path, "w") as conf:
                    conf.write(simple_blueconfig)
                circuit = Circuit(config_path)
                assert circuit.morph.get_filepath(1) == os.path.join(TEST_DATA_DIR, expected_morph)

        _test_morphs("h5", "morph-A.h5")
        _test_morphs("asc", "morph-A.asc")
        _test_morphs("swc", "morph-A.swc")
        # even if these ones are not part of blueconfig specs, they are part of the historical
        # bluepy dispatch
        _test_morphs("h5v1", "morph-A.h5")
        _test_morphs("ascii", "morph-A.asc")
        with pytest.raises(BluePyError):
            _test_morphs("unknown", "morph-A.asc")
