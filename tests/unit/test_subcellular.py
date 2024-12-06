import os
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepy.subcellular as test_module
from bluepy import Synapse
from bluepy.subcellular import Organelle

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
TEST_FILE = os.path.join(TEST_DATA_DIR, "subcellular.h5")


class TestSubcellularHelper:
    def setup_method(self):
        circuit = Mock()
        self.test_obj = test_module.SubcellularHelper(
            TEST_FILE, circuit
        )

    def test_gene_mapping_1(self):
        actual = self.test_obj.gene_mapping()
        pdt.assert_frame_equal(
            actual,
            pd.DataFrame(
                [
                    ['protA', 'protA;protC', 'Foo'],
                    ['protB', 'protB;protC', ''],
                ],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                columns=['lead_protein', 'maj_protein', 'comment']
            )
        )

    def test_gene_mapping_2(self):
        actual = self.test_obj.gene_mapping('gene1')
        pdt.assert_series_equal(
            actual,
            pd.Series(
                ['protA', 'protA;protC', 'Foo'],
                index=['lead_protein', 'maj_protein', 'comment'],
                name='gene1'
            )
        )

    def test_gene_expressions_1(self):
        actual = self.test_obj.gene_expressions(1)
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [1.0, 2.0],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name='expr'
            ),
            check_dtype=False
        )

    def test_gene_expressions_2(self):
        actual = self.test_obj.gene_expressions(1, 'gene1')
        assert actual == 1.0

    def test_cell_proteins_1(self):
        actual = self.test_obj.cell_proteins(1, organelles=[Organelle.TOTAL, Organelle.NUCLEUS])
        pdt.assert_frame_equal(
            actual,
            pd.DataFrame(
                [
                    [100., 2.],
                    [200., 4.],
                ],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                columns=[Organelle.TOTAL, Organelle.NUCLEUS]
            ),
            check_dtype=False
        )

    def test_cell_proteins_2(self):
        actual = self.test_obj.cell_proteins(1, organelles=Organelle.TOTAL)
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [100., 200.],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name=Organelle.TOTAL
            ),
            check_dtype=False
        )

    def test_cell_proteins_3(self):
        actual = self.test_obj.cell_proteins(1, Organelle.TOTAL, 'gene1')
        assert actual == 100.0

    def test_synapse_proteins_1(self):
        self.test_obj._circuit.connectome.synapse_properties.return_value = pd.DataFrame({
            Synapse.POST_GID: [1],
        })
        actual = self.test_obj.synapse_proteins((42, 1), 'pre')
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [np.nan, 1000.],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name='pre'
            ),
            check_dtype=False
        )

    def test_synapse_proteins_2(self):
        self.test_obj._circuit.connectome.synapse_properties.return_value = pd.DataFrame({
            Synapse.POST_GID: [1],
        })
        actual = self.test_obj.synapse_proteins((42, 1), 'pre', 'gene2')
        assert actual == 1000.

    def test_synapse_proteins_3(self):
        self.test_obj._circuit.connectome.synapse_properties.return_value = pd.DataFrame({
            Synapse.POST_GID: [1],
            Synapse.TYPE: [101],
            Synapse.G_SYNX: [10.],
        })
        actual = self.test_obj.synapse_proteins((42, 1), 'post')
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [1.2, 3.6],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name='post_exc'
            ),
            check_dtype=False
        )

    def test_synapse_proteins_4(self):
        self.test_obj._circuit.connectome.synapse_properties.return_value = pd.DataFrame({
            Synapse.POST_GID: [1],
            Synapse.TYPE: [1],
            Synapse.G_SYNX: [10.],
        })
        actual = self.test_obj.synapse_proteins((42, 1), 'post')
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [1.42, 2.84],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name='post_inh'
            ),
            check_dtype=False
        )

    def test_synapse_proteins_5(self):
        self.test_obj._circuit.connectome.synapse_properties.return_value = pd.DataFrame({
            Synapse.POST_GID: [1],
            Synapse.TYPE: [101],
            Synapse.G_SYNX: [10.],
        })
        actual = self.test_obj.synapse_proteins((42, 1), 'post', area_per_nS=11.)
        pdt.assert_series_equal(
            actual,
            pd.Series(
                [110., 330.],
                index=pd.Index(['gene1', 'gene2'], name='gene'),
                name='post_exc'
            ),
            check_dtype=False
        )

    def test_synapse_proteins_raises_1(self):
        with pytest.raises(ValueError):
            self.test_obj.synapse_proteins((42, 1), 'foo')
