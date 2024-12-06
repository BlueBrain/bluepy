from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from morphio import SectionType

import bluepy.circuit_stats as test_module
from bluepy.enums import Cell, Section, Segment, Synapse
from bluepy.exceptions import BluePyError
from bluepy.geometry.roi import Cube


class TestCircuitStatsHelper:
    def setup_method(self):
        self.circuit = Mock()
        self.stats = test_module.StatsHelper(self.circuit)

    def test_cell_density(self):
        self.circuit.cells.get.return_value = pd.DataFrame([
            [0., 0., 0.],  # included
            [0., 0., 0.],  # excluded by `group`
            [1., 0., 0.],  # excluded by `roi.contains`
        ],
            index=[1, 2, 3],
            columns=[Cell.X, Cell.Y, Cell.Z]
        )
        self.circuit.cells.ids.return_value = [1, 3, 9999]
        roi = Cube((0, 0, 0), 1)
        actual = self.stats.cell_density(roi, group='test')
        npt.assert_almost_equal(actual, 1e9)

    def test_synapse_density(self):
        self.circuit.connectome.spatial_index.q_window.return_value = pd.DataFrame([
            [0., 0., 0., 0., 0., 0.],  # included
            [1., 0., 0., 1., 0., 0.],  # excluded by `roi.contains`
        ],
            columns=[
                Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER, Synapse.PRE_Z_CENTER,
                Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER
            ]
        )
        roi = Cube((0, 0, 0), 1)
        actual = self.stats.synapse_density(roi)
        assert actual == 1e9

    def test_fibre_density(self):
        self.circuit.morph.spatial_index.q_window.return_value = pd.DataFrame([
            [0.0, 0., 0., 1., 0.1, 0., 0., 1., SectionType.axon],  # included
            [0.0, 0., 0., 1., 0.1, 0., 0., 1., SectionType.soma],  # excluded by neurite type
            [1.0, 0., 0., 1., 0.1, 0., 0., 1., SectionType.axon],
            # excluded by `roi.contains` (segment start point)
            [0.0, 0., 0., 1., 1.0, 0., 0., 1., SectionType.axon],
            # excluded by `roi.contains` (segment end point)
        ],
            columns=[
                Segment.X1, Segment.Y1, Segment.Z1, Segment.R1,
                Segment.X2, Segment.Y2, Segment.Z2, Segment.R2,
                Section.NEURITE_TYPE
            ]
        )
        roi = Cube((0, 0, 0), 1)
        actual = self.stats.fibre_density(roi, SectionType.axon)
        assert actual == 0.1 * np.pi

    def test_divergence_1(self):
        self.circuit.cells.ids.return_value = [1, 2]
        self.circuit.connectome.iter_connections.return_value = [(1, None, 42), (1, None, 43)]
        actual = self.stats.sample_divergence('pre', 'post', by='syn')
        npt.assert_equal(actual, [85, 0])

    def test_divergence_2(self):
        self.circuit.cells.ids.return_value = [1, 2]
        self.circuit.connectome.iter_connections.return_value = [(1, None), (1, None)]
        actual = self.stats.sample_divergence('pre', 'post', by='conn')
        npt.assert_equal(actual, [2, 0])

    def test_divergence_3(self):
        pytest.raises(BluePyError, self.stats.sample_divergence, 'pre', 'post', by='err')

    def test_convergence_1(self):
        self.circuit.cells.ids.return_value = [1, 2]
        self.circuit.connectome.iter_connections.return_value = [(None, 2, 42), (None, 2, 43)]
        actual = self.stats.sample_convergence('pre', 'post', by='syn')
        npt.assert_equal(actual, [0, 85])

    def test_convergence_2(self):
        self.circuit.cells.ids.return_value = [1, 2]
        self.circuit.connectome.iter_connections.return_value = [(None, 2), (None, 2)]
        actual = self.stats.sample_convergence('pre', 'post', by='conn')
        npt.assert_equal(actual, [0, 2])

    def test_convergence_3(self):
        pytest.raises(BluePyError, self.stats.sample_convergence, 'pre', 'post', by='err')

    def test_synapse_region_distribution_1(self):
        self.circuit.connectome.pathway_synapses.return_value = pd.DataFrame([
            [1, 11, None, None],
            [3, 22, None, None],
            [2, 22, None, None],
            [4, 11, None, None],
        ],
            columns=[Synapse.PRE_GID, Synapse.PRE_X_CENTER, Synapse.PRE_Y_CENTER,
                     Synapse.PRE_Z_CENTER]
        )
        self.circuit.cells.get.return_value = pd.Series(
            ['mtype-A', 'mtype-B', 'mtype-A', 'mtype-A'],
            index=[1, 2, 3, 4]
        )
        atlas = Mock()
        atlas.lookup = lambda x, outer_value: x[:, 0]
        actual = self.stats.synapse_region_distribution(atlas, 'pre', Cell.MTYPE, 'pre', 'post')
        expected = pd.DataFrame([
            [2, 1],
            [0, 1],
        ],
            index=pd.Index(['mtype-A', 'mtype-B'], name=Cell.MTYPE),
            columns=pd.Index([11, 22], name='region')
        )
        pdt.assert_frame_equal(actual, expected)

    def test_bouton_density_1(self):
        section = Mock()
        section.type = SectionType.axon
        section.points = np.array([[0, 0, 0], [10, 0, 0]])  # --> axon_length should be 10
        morph = Mock()
        morph.sections = [section]
        self.circuit.morph.get.return_value = morph
        self.circuit.connectome.efferent_synapses.return_value = [(1, 0), (2, 0), (2, 1)]
        actual = self.stats.bouton_density(42, synapses_per_bouton=1.5)
        npt.assert_almost_equal(actual, 0.2)

    def test_bouton_density_2(self):
        mock_region_mask = Mock()
        mock_region_mask.lookup.return_value = np.array([False])
        self.circuit.atlas.get_region_mask.return_value = mock_region_mask
        self.circuit.morph.segment_points.return_value = pd.DataFrame(
            data=[
                [0., 1., 1., 0., 2., 2.],  # both endpoints out of target region
            ],
            columns=[
                Segment.X1, Segment.Y1, Segment.Z1,
                Segment.X2, Segment.Y2, Segment.Z2,
            ]
        )
        actual = self.stats.bouton_density(42, region='Foo')
        assert np.isnan(actual)

    def test_bouton_density_3(self):
        def _mock_lookup(points, outer_value):
            return np.all(points > 0, axis=-1)

        mock_region_mask = Mock()
        mock_region_mask.lookup.side_effect = _mock_lookup
        self.circuit.atlas.get_region_mask.return_value = mock_region_mask
        self.circuit.morph.segment_points.return_value = pd.DataFrame(
            data=[
                [0., 0., 0., 1., 1., 1.],  # first endpoint out of target region
                [1., 1., 1., 3., 3., 3.],  # both endpoints within target region
                [1., 1., 1., 4., 4., 4.],  # both endpoints within target region
                [1., 1., 1., 5., 5., 5.],  # both endpoints within target region
                [1., 1., 1., 0., 0., 0.],  # second endpoint out of target region
            ],
            columns=[
                Segment.X1, Segment.Y1, Segment.Z1,
                Segment.X2, Segment.Y2, Segment.Z2,
            ],
            index=pd.MultiIndex.from_tuples([
                (11, 0),  # "outer" segment
                (11, 1),  # "inner" segment
                (11, 2),  # "inner" segment
                (12, 0),  # "inner" segment
                (12, 1),  # "outer" segment
            ])
        )
        self.circuit.connectome.efferent_synapses.return_value = pd.DataFrame(
            data=[
                [11, 0],  # "outer" segment
                [11, 1],  # "inner" segment
                [12, 0],  # "inner" segment
                [11, 1],  # "inner" segment
                [11, 1],  # "inner" segment
            ],
            columns=[
                Synapse.PRE_SECTION_ID,
                '_PRE_SEGMENT_ID'
            ]
        )
        expected = (
                           3 + 1  # three synapses on (11, 1); one synapse on (12, 0)
                   ) / sum([
            np.sqrt(12),  # length((11, 1))
            np.sqrt(27),  # length((11, 2))
            np.sqrt(48),  # length((12, 0))
        ])
        actual = self.stats.bouton_density(42, region='Foo')
        npt.assert_almost_equal(actual, expected)

    @patch.object(test_module.StatsHelper, '_calc_bouton_density')
    def test_sample_bouton_density(self, mock_get):
        mock_get.side_effect = [42., 43.]
        self.circuit.cells.ids.return_value = [1, 2, 3]
        npt.assert_equal(
            self.stats.sample_bouton_density(2),
            [42., 43.]
        )

    def test_sample_pathway_synapse_count(self):
        self.circuit.connectome.iter_connections.return_value = [
            (0, 0, 42), (0, 0, 43), (0, 0, 44)
        ]
        npt.assert_equal(
            self.stats.sample_pathway_synapse_count(2),
            [42, 43]
        )

    def test_segment_region_distribution(self):
        section_apical_dendrite = Mock()
        section_apical_dendrite.type = SectionType.apical_dendrite
        section_apical_dendrite.points = np.array([[0., 0., 0.], [1., 1., 1.]])
        section_apical_dendrite.diameters = np.array([1., 1.])

        section_axon = Mock()
        section_axon.type = SectionType.axon
        section_axon.points = np.array([[0.5, 0.5, 1.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [5., 5., 5.]])
        section_axon.diameters = np.array([1., 1., 1., 1., 2.])

        section_basal_dendrite = Mock()
        section_basal_dendrite.type = SectionType.basal_dendrite
        section_basal_dendrite.points = np.array([[],])
        section_basal_dendrite.diameters = np.array([[],])

        morph = Mock()
        morph.sections = [section_apical_dendrite, section_axon, section_basal_dendrite]

        self.circuit.morph.get.return_value = morph
        self.circuit.cells.ids.return_value = [1]

        def _mock_lookup(positions, outer_value=None):
            return positions[:, 2].astype(int)

        mock_region_mask = Mock()
        mock_region_mask.lookup.side_effect = _mock_lookup
        tested = self.stats.segment_region_distribution(mock_region_mask, 1)
        expected = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [0, 2, 1, 1],
                [0, 0, 0, 0],
            ],
            columns=[0, 1, 2, 4],
            index=pd.MultiIndex.from_tuples(
                [(1, 'apical_dendrite'), (1, 'axon'), (1, 'basal_dendrite')],
                names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.stats.segment_region_distribution(mock_region_mask, 1, normalize=True)
        expected = pd.DataFrame(
            [
                [1., 0., 0., 0.],
                [0., 0.5, 0.25, 0.25],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            columns=[0, 1, 2, 4],
            index=pd.MultiIndex.from_tuples(
                [(1, 'apical_dendrite'), (1, 'axon'), (1, 'basal_dendrite')],
                names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.stats.segment_region_distribution(mock_region_mask, 1, by='length')
        expected = pd.DataFrame(
            [
                [np.sqrt(3), 0., 0., 0.],
                [0., np.sqrt(3) + np.linalg.norm(np.array([0.5, 0.5, 1.]) - np.array([1., 1., 1.])), np.sqrt(3), np.sqrt(3)*2],
                [0, 0, 0, 0],
            ],
            columns=[0, 1, 2, 4],
            index=pd.MultiIndex.from_tuples(
                [(1, 'apical_dendrite'), (1, 'axon'), (1, 'basal_dendrite')],
                names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.stats.segment_region_distribution(mock_region_mask, 1, by='volume')
        expected = pd.DataFrame(
            [
                [1.36035, 0., 0., 0.],
                [0., 1.91571, 1.36035, 6.348298],
                [0, 0, 0, 0],
            ],
            columns=[0, 1, 2, 4],
            index=pd.MultiIndex.from_tuples(
                [(1, 'apical_dendrite'), (1, 'axon'), (1, 'basal_dendrite')],
                names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(tested, expected)
        with pytest.raises(ValueError):
            self.stats.segment_region_distribution(mock_region_mask, 1, by='err')
