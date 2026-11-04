"""The algorithm on a small tree whose results were worked out by hand."""
import numpy as np
import pytest

from ctreex import TreeCounterfactualExplainer
from ctreex.counterfactuals import (describe_range, grow_region,
                                    support_set_metric, volatility_index)
from ctreex.distances import categorical_distance, exact_distance


@pytest.fixture
def explainer(toy_tree, toy_spec):
    return TreeCounterfactualExplainer(toy_tree, toy_spec)


def test_paths_and_meta_features(explainer):
    assert [p.leaf for p in explainer.paths] == [2, 3, 5, 6]
    assert explainer.paths[0].conditions == [('age', '<=', 30.),
                                             ('job_a', '=', 0)]
    assert [m.name for m in explainer.meta_features] == [
        'age<=30.0', '30.0<age<=50.0', '50.0<age', 'job_a']
    lmf = explainer.leaf_meta_features
    assert lmf.matrix.tolist() == [[1, -1, -1, -1], [1, -1, -1, 1],
                                   [-1, 1, -1, 0], [-1, -1, 1, 0]]
    assert lmf.leaf_classes.tolist() == ['no', 'yes', 'yes', 'no']
    assert [(kind, feature) for kind, feature, _ in lmf.groups] == [
        ('categorical', 'job'), ('numerical', 'age')]


def test_explanation(explainer, toy_spec):
    x = toy_spec.encode({'age': 25, 'job': 'b'})
    e = explainer.explain_instance(x)
    assert (e.leaf, e.prediction) == (2, 'no')
    assert e.meta_features.tolist() == [1, -1, -1, -1]
    assert e.leaf_distances.tolist() == [0, 1, 1, 1]
    assert e.support == {0: {'same': [2], 'other': []},
                         1: {'same': [6], 'other': [3, 5]}}
    found = e.at_distance(1)
    assert found.per_feature == {'job': [3], 'age': [5]}
    assert found.ranked == {
        0: {'job': {'is': ['b'], 'wannabe': ['a']}},
        1: {'age': {'is': [25], 'wannabe': [['30.0', '50.0']]}}}
    assert explainer.describe(found.ranked) == {
        0: {'job type': {'is': ['baker'], 'wannabe': ['artist']}},
        1: {'age': {'is': [25], 'wannabe': ['between 30.0 and 50.0']}}}


def test_excluded_and_included(explainer, toy_spec):
    x = toy_spec.encode({'age': 25, 'job': 'b'})
    assert explainer.explain_instance(x, excluded=['age']).at_distance(
        1).per_feature == {'job': [3]}
    assert explainer.explain_instance(x, included=['age']).at_distance(
        1).per_feature == {'age': [5]}


def test_leaf_distance_matrix(explainer):
    matrix, ids, _ = explainer.leaf_distance_matrix()
    assert ids.tolist() == [2, 3, 5, 6]
    assert matrix[0, 1] == 1      # job: one-hot vectors disagree
    assert matrix[0, 2] == 0.5    # age: 2 of 4 meta-features disagree
    assert np.allclose(matrix, matrix.T)


def test_volatility_and_support():
    matrix = np.array([[0., 1, 2], [1, 0, 1], [2, 1, 0]])
    ids, classes = np.array([0, 1, 2]), np.array(['a', 'a', 'b'])
    # bins [0, .5] (.5, 1] (1, 1.5] (1.5, 2]; ratios 1/2, 1/2, 1, 2
    assert volatility_index(0, matrix, ids, classes) == pytest.approx(
        0.5 + 0.5 / 2 + 1 / 3 + 2 / 4)
    assert support_set_metric(0, matrix, ids, classes) == {
        '(-0.0001, 0.5]': {'same': [0], 'other': []},
        '(0.5, 1.0]': {'same': [1], 'other': []},
        '(1.0, 1.5]': {'same': [], 'other': []},
        '(1.5, 2.0]': {'same': [], 'other': [2]}}


def test_building_blocks():
    assert exact_distance(np.array([0, 1]), np.array([1, -1])) == 0
    assert exact_distance(np.array([-1, 1]), np.array([1, -1])) == 1
    assert exact_distance(np.array([1, 0]), np.array([-1, -1])) == 1
    with pytest.raises(ValueError):
        exact_distance(np.array([1, 0]), np.array([1, 1]))
    assert categorical_distance(np.array([1, -1, -1]),
                                np.array([-1, 0, 0])) == 1
    assert categorical_distance(np.array([0, 0]), np.array([1, -1])) == 0
    assert grow_region(['1', '2'], [('2', '3'), ('0', '1')]) == \
        (['0', '3'], [])
    assert describe_range([None, '5']) == 'less or equal to 5'
    assert describe_range(['5', None]) == 'more than 5'
    with pytest.raises(ValueError):
        describe_range([None, None])


def test_plots(explainer):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from ctreex.plot import (leaf_metric_barplot, leaf_metric_histogram,
                             metric_heat_map)
    matrix, ids, classes = explainer.leaf_distance_matrix()
    assert metric_heat_map(matrix, ids, classes, annot=True, xtick=True,
                           ytick=True) is not None
    assert leaf_metric_barplot(2, matrix, ids, classes) is not None
    assert leaf_metric_histogram(2, matrix, ids, classes) is not None
