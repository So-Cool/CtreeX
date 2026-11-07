"""The algorithm on a small tree whose results were worked out by hand."""
import numpy as np
import pytest

from ctreex import TreeCounterfactualExplainer, TreeStructure
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
    # every category of a categorical feature used by the tree is described,
    # also those the tree never splits on (job_b and job_c)
    assert [m.name for m in explainer.meta_features] == [
        'age<=30.0', '30.0<age<=50.0', '50.0<age', 'job_a', 'job_b', 'job_c']
    lmf = explainer.leaf_meta_features
    assert lmf.matrix.tolist() == [
        [1, -1, -1, -1, 0, 0], [1, -1, -1, 1, -1, -1],
        [-1, 1, -1, 0, 0, 0], [-1, -1, 1, 0, 0, 0]]
    assert lmf.leaf_classes.tolist() == ['no', 'yes', 'yes', 'no']
    assert [(kind, feature) for kind, feature, _ in lmf.groups] == [
        ('categorical', 'job'), ('numerical', 'age')]


def test_single_threshold_intervals():
    tree = TreeStructure(
        children_left=[1, -1, -1], children_right=[2, -1, -1],
        feature=[0, -2, -2], threshold=[0.5, -2, -2],
        value=[[1, 1], [1, 0], [0, 1]], classes=['n', 'y'])
    explainer = TreeCounterfactualExplainer(tree, ['x'])
    # a feature split at a single threshold has two intervals
    assert [m.name for m in explainer.meta_features] == ['x<=0.5', '0.5<x']
    assert explainer.leaf_meta_features.matrix.tolist() == [[1, -1],
                                                            [-1, 1]]


def test_explanation(explainer, toy_spec):
    x = toy_spec.encode({'age': 25, 'job': 'b'})
    e = explainer.explain_instance(x)
    assert (e.leaf, e.prediction) == (2, 'no')
    assert e.meta_features.tolist() == [1, -1, -1, -1, 1, -1]
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
    # age: 2 of the 6 meta-features disagree
    assert matrix[0, 2] == pytest.approx(2 / 6)
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


def test_integer_ranges(toy_tree):
    from ctreex import Feature, FeatureSpec
    from ctreex.counterfactuals import describe_integer_range
    assert describe_integer_range([None, '883.0']) == 'at most 883'
    assert describe_integer_range(['954.5', '1181.5']) == \
        'between 955 and 1181'
    assert describe_integer_range(['22.5', None]) == 'at least 23'
    assert describe_integer_range(('22.5', '23.5')) == 'exactly 23'
    assert describe_integer_range(['22.1', '22.9']) == \
        'between 22.1 and 22.9'  # no whole number inside
    spec = FeatureSpec.from_column_names(
        ['age', 'job_a', 'job_b', 'job_c'],
        features=[Feature('age', 'numerical', integer=True)])
    explainer = TreeCounterfactualExplainer(toy_tree, spec)
    ranked = explainer.explain_instance(
        spec.encode({'age': 25, 'job': 'b'})).at_distance(1).ranked
    assert explainer.describe(ranked)[1]['age']['wannabe'] == \
        ['between 30.0 and 50.0']
    assert explainer.describe(ranked, integer_ranges=True)[1]['age'][
        'wannabe'] == ['between 31 and 50']
    with pytest.raises(ValueError):
        Feature('job', 'categorical', integer=True)


def test_short_thresholds():
    from ctreex.counterfactuals import describe_range, format_bound
    assert format_bound('0.800000011920929', short=True) == '0.8'
    assert format_bound('4.950000047683716', short=True) == '4.95'
    for kept in ('954.5', '883.0', '0.1', '2250.0', '1e+20'):
        assert format_bound(kept, short=True) == kept
    assert format_bound('0.800000011920929') == '0.800000011920929'
    assert describe_range(['0.800000011920929', '1.75'], short=True) == \
        'between 0.8 and 1.75'
    # the shortened bound selects the same 32-bit data as the threshold
    for text in ('0.800000011920929', '4.950000047683716',
                 '4.8500001430511475'):
        threshold = float(text)
        short = np.float32(float(format_bound(text, short=True)))
        below = np.nextafter(short, np.float32(np.inf))
        assert float(short) <= threshold < float(below)


def test_short_thresholds_in_descriptions():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    X = np.array([[0.6], [0.7], [1.0], [1.2]], dtype=np.float32)
    clf = sklearn_tree.DecisionTreeClassifier(random_state=0).fit(
        X, ['a', 'a', 'b', 'b'])
    # the threshold, 0.8499999940395355, lies just below the 32-bit 0.85
    # (0.85 goes right), hence the largest 32-bit number below it is shown
    explainer = TreeCounterfactualExplainer(clf, ['x'])
    found = explainer.explain_instance(np.array([0.6])).counterfactuals[0]
    assert explainer.describe(found.ranked)[0]['x']['wannabe'] == [
        'more than 0.84999996']
    # the structured counterfactuals keep the exact threshold
    assert found.ranked[0]['x']['wannabe'] == [['0.8499999940395355', None]]
    assert clf.apply(np.array([[0.85]], dtype=np.float32))[0] == \
        clf.apply(np.array([[1.2]], dtype=np.float32))[0]


def test_class_equivalent(explainer, toy_spec):
    x = toy_spec.encode({'age': 25, 'job': 'b'})  # leaf 2 ('no')
    e = explainer.explain_instance(x, counterfactual=False)
    assert e.prediction == 'no' and not e.counterfactual
    # leaf 6 ('no') is reached by changing the age only
    assert e.distances == [1]
    found = e.at_distance(1)
    assert found.per_leaf == {6: {'age': {'is': [25],
                                          'wannabe': [['50.0', None]]}}}
    assert found.leaf_classes == {6: 'no'}
    assert explainer.describe(found.ranked) == {
        0: {'age': {'is': [25], 'wannabe': ['more than 50.0']}}}
    # the default stays class-contrastive
    assert explainer.explain_instance(x).counterfactual
    with pytest.raises(ValueError):
        explainer.explain_instance(x, counterfactual=False,
                                   target_class='yes')


def test_class_equivalent_keeps_the_prediction(explainer, toy_spec):
    for age in (20, 35, 60):
        for job in 'abc':
            values = {'age': age, 'job': job}
            x = toy_spec.encode(values)
            e = explainer.explain_instance(x, counterfactual=False)
            for found in e.counterfactuals:
                # (empty at distances with leaves of the other class only)
                assert set(found.leaf_classes.values()) <= {e.prediction}
                for leaf in found.per_leaf:
                    assert leaf != e.leaf
