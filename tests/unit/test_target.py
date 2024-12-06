import os

import numpy as np
import numpy.testing as npt
import pytest

import bluepy.impl.target as test_module
from bluepy.exceptions import BluePyError

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_target_init_success():
    test_module.Target(filename="filename", name="Q", type_="Cell", start=10, end=20)


def test_target_init_failure():
    with pytest.raises(BluePyError, match="Unknown target type: UnknownType"):
        test_module.Target(filename="filename", name="Q", type_="UnknownType", start=10, end=20)


def test_target_context():
    context = test_module.TargetContext.load([
        os.path.join(TEST_DATA_DIR, "A.target"),
        os.path.join(TEST_DATA_DIR, "C.target"),
    ])
    assert context.names == ['Empty', 'Q', 'X', 'Y', 'Z']
    npt.assert_equal(context.resolve('Empty'), [])
    npt.assert_equal(context.resolve('Z'), [1, 2, 3, 4])
    npt.assert_equal(context.resolve('Z'), [1, 2, 3, 4])  # check caching
    pytest.raises(BluePyError, context.resolve, 'NoSuchTarget')


def test_target_context_duplicate():
    pytest.raises(BluePyError, test_module.TargetContext.load, [
        os.path.join(TEST_DATA_DIR, "A.target"),
        os.path.join(TEST_DATA_DIR, "A.target"),
    ])


def test_to_node_set():
    context = test_module.TargetContext.load([
        os.path.join(TEST_DATA_DIR, "A.target"),
        os.path.join(TEST_DATA_DIR, "C.target"),
        os.path.join(TEST_DATA_DIR, "D.target"),
    ])

    def _compared(dict_tested, dict_expected):
        for key in dict_tested:
            npt.assert_equal(dict_tested[key]['node_id'], dict_expected[key]['node_id'])
            if 'population' in dict_tested[key]:
                assert dict_tested[key]['population'] == dict_expected[key]['population']

    tested = context.to_node_sets()
    expected = {'Y': {'node_id': np.array([2, 3])},
                'X': {'node_id': np.array([0, 1])},
                'Q': {'node_id': np.array([])},
                'Empty': {'node_id': np.array([])},
                'Z': {'node_id': np.array([0, 1, 2, 3])},
                'pop_name:Z': {'node_id': np.array([0, 1, 2])}
                }
    _compared(tested, expected)

    tested = context.to_node_sets(one_based=True)
    expected = {'Y': {'node_id': np.array([3, 4])},
                'X': {'node_id': np.array([1, 2])},
                'Q': {'node_id': np.array([])},
                'Empty': {'node_id': np.array([])},
                'Z': {'node_id': np.array([1, 2, 3, 4])},
                'pop_name:Z': {'node_id': np.array([1, 2, 3])}
                }
    _compared(tested, expected)


def test_section_targets():
    context = test_module.TargetContext.load([os.path.join(TEST_DATA_DIR, "section.target")])
    npt.assert_equal(context.resolve("single_cell"), [1])
    npt.assert_equal(context.resolve("mosaic_soma"), [1, 2])
    npt.assert_equal(context.resolve("mosaic_axon"), [1, 2])
    npt.assert_equal(context.resolve("axon_comp_mosaic"), [1, 2])
    npt.assert_equal(context.resolve('axon_comp_mosaic_others'), [1, 2, 3, 4])
    npt.assert_equal(context.resolve('mosaic_other_soma'), [1, 2, 3, 4])
    npt.assert_equal(context.resolve('two_cells_soma'), [1, 2])
    npt.assert_equal(context.resolve('mosaic_3_soma'), [1, 2, 3])

    # can only have a single type of section
    with pytest.raises(BluePyError):
        context.resolve('mosaic_axon_soma')

    # target can not have just a single section type with no target
    with pytest.raises(BluePyError):
        context.resolve('soma')


def test_filter():
    context = test_module.TargetContext.load([
        os.path.join(TEST_DATA_DIR, "colon.target"),])
    assert set(context.names) == {'A', 'default2:A', 'default:A'}
    tested = context.filter({'A', 'default2:A'}, inplace=False)
    assert set(tested.names) == {'A', 'default2:A'}
    context.filter({'A', 'default:A'}, inplace=True)
    assert set(context.names) == {'A', 'default:A'}
