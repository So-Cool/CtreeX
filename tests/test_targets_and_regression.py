"""Regression trees, class probabilities and several target classes."""
import json

import numpy as np
import pytest

from ctreex import (ClassProbability, OtherClass, SameClass, TargetClass,
                    TreeCounterfactualExplainer, TreeStructure, ValueAtLeast,
                    ValueAtMost, ValueChange, ValueWithin, load_tree)


def _three_leaves(value, classes=None):
    """x <= 10 -> leaf 1; 10 < x <= 20 -> leaf 3; x > 20 -> leaf 4."""
    return TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 0, -2, -2], threshold=[10., -2, 20., -2, -2],
        value=value, classes=classes, feature_names=['x'])


@pytest.fixture
def regression_tree():
    return _three_leaves([[150.], [100.], [200.], [150.], [300.]])


@pytest.fixture
def probability_tree():
    # leaf 1: 90% a; leaf 3: 60% a; leaf 4: 80% b
    return _three_leaves([[5, 5], [9, 1], [4, 6], [6, 4], [2, 8]],
                         classes=['a', 'b'])


def _described(explainer, explanation):
    return [(d, values, explainer.describe({0: cf})[0]['x']['wannabe'])
            for d, values, cf in explanation.ranked(with_values=True)]


def test_regression_tree_structure(regression_tree):
    tree = regression_tree
    assert tree.task == 'regression' and tree.classes is None
    X = np.array([[5.], [15.], [25.]])
    assert tree.predict(X).tolist() == [100., 150., 300.]
    assert tree.leaf_value(4) == 300.
    with pytest.raises(TypeError):
        tree.predict_proba(X)
    with pytest.raises(TypeError):
        tree.leaf_class(4)
    with pytest.raises(ValueError):  # a regression tree has one value
        TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2], [0.5, -2, -2],
                      [[1, 2], [1, 0], [0, 2]])
    assert 'task=regression' in repr(tree)
    copy = load_tree(json.loads(tree.to_json()))
    assert copy.task == 'regression'
    assert np.array_equal(copy.predict(X), tree.predict(X))


def test_tree_format_versions(toy_tree):
    d = toy_tree.to_dict()
    assert d['format_version'] == 2 and d['task'] == 'classification'
    # version 1 had no task -- classifiers only
    old = {k: v for k, v in d.items() if k != 'task'}
    old['format_version'] = 1
    assert load_tree(old).task == 'classification'
    with pytest.raises(ValueError):
        load_tree(dict(d, task='clustering'))


def test_sklearn_regressor():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.uniform(0, 100, 300),
                         np.eye(2)[rng.randint(0, 2, 300)]])
    y = 10 * X[:, 0] + 500 * X[:, 2] + rng.normal(0, 5, 300)
    reg = sklearn_tree.DecisionTreeRegressor(max_depth=4,
                                             random_state=0).fit(X, y)
    tree = TreeStructure.from_sklearn(reg)
    assert tree.task == 'regression'
    assert np.allclose(tree.predict(X), reg.predict(X))
    explainer = TreeCounterfactualExplainer(reg, ['size', 'garden_no',
                                                  'garden_yes'])
    ids = explainer.leaf_meta_features.leaf_ids.tolist()
    values = explainer.leaf_meta_features.leaf_values
    for x in X[:20]:
        prediction = float(reg.predict(x.reshape(1, -1))[0])
        for target, keep in (
                (ValueChange(+200), lambda v: v >= prediction + 200),
                (ValueChange(-200), lambda v: v <= prediction - 200),
                (ValueAtLeast(800), lambda v: v >= 800),
                (ValueAtMost(300), lambda v: v <= 300),
                (ValueWithin(50), lambda v: abs(v - prediction) <= 50)):
            e = explainer.explain_instance(x, target=target)
            assert e.prediction == pytest.approx(prediction)
            assert e.task == 'regression'
            for found in e.counterfactuals:
                for leaf, value in found.leaf_values.items():
                    assert keep(value) and leaf != e.leaf
                    assert value == values[ids.index(leaf)]
                for rank, (low, high) in found.ranked_values.items():
                    assert keep(low) and keep(high)


def test_regression_counterfactuals(regression_tree):
    explainer = TreeCounterfactualExplainer(regression_tree)
    x = np.array([5.])  # leaf 1, 100
    e = explainer.explain_instance(x, target=ValueChange(+100))
    assert e.prediction == 100. and e.target == ValueChange(100)
    assert _described(explainer, e) == [
        (1, (300., 300.), ['more than 20.0'])]
    # leaves of different values are not merged
    e = explainer.explain_instance(x, target=ValueChange(+50))
    assert sorted(_described(explainer, e)) == [
        (1, (150., 150.), ['between 10.0 and 20.0']),
        (1, (300., 300.), ['more than 20.0'])]
    assert e.at_distance(1).leaf_values == {3: 150., 4: 300.}
    assert e.at_distance(1).leaf_classes == {}
    # the changes that keep the prediction within 60
    e = explainer.explain_instance(x, target=ValueWithin(60))
    assert _described(explainer, e) == [
        (1, (150., 150.), ['between 10.0 and 20.0'])]
    # the leaf of the data point is never used
    assert all(not found.per_leaf for found in explainer.explain_instance(
        x, target=ValueAtMost(100)).counterfactuals)
    e = explainer.explain_instance(np.array([25.]), target=ValueChange(-120))
    assert sorted(_described(explainer, e)) == [
        (1, (100., 100.), ['less or equal to 10.0']),
        (1, (150., 150.), ['between 10.0 and 20.0'])]
    with pytest.raises(ValueError):  # the classes of a regression tree
        e.ranked(with_classes=True)


def test_regression_targets_are_required(regression_tree):
    explainer = TreeCounterfactualExplainer(regression_tree)
    x = np.array([5.])
    with pytest.raises(ValueError):
        explainer.explain_instance(x)
    for target in (OtherClass(), SameClass(), TargetClass('a'),
                   ClassProbability('a', at_least=0.5)):
        with pytest.raises(ValueError):
            explainer.explain_instance(x, target=target)
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(regression_tree, class_names=['a'])
    for invalid in (lambda: ValueChange(0), lambda: ValueWithin(-1),
                    lambda: ValueAtLeast(float('nan')),
                    lambda: ValueChange('100')):
        with pytest.raises(ValueError):
            invalid()


def test_regression_targets_need_regression_trees(probability_tree):
    explainer = TreeCounterfactualExplainer(probability_tree)
    for target in (ValueChange(1), ValueAtLeast(1), ValueAtMost(1),
                   ValueWithin(1)):
        with pytest.raises(ValueError):
            explainer.explain_instance(np.array([5.]), target=target)
    with pytest.raises(ValueError):  # the values of a classification tree
        explainer.explain_instance(np.array([5.])).ranked(with_values=True)


def test_regression_leaf_distances(regression_tree):
    explainer = TreeCounterfactualExplainer(regression_tree)
    matrix, ids, values = explainer.leaf_distance_matrix()
    assert values.tolist() == [100., 150., 300.]
    with pytest.raises(ValueError):  # regression trees need a tolerance
        explainer.volatility_index(1)
    # within 50: 100 and 150 are the same, 300 is other
    assert explainer.support_set_metric(1, tolerance=50) == {
        0: {'same': [1], 'other': []},
        1: {'same': [3], 'other': [4]}}
    assert explainer.volatility_index(1, tolerance=50) == pytest.approx(
        (1 + 1) / (1 * (1 + 1)))
    assert explainer.support_set_metric(1, tolerance=0)[1] == {
        'same': [], 'other': [3, 4]}
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from ctreex.plot import (leaf_metric_barplot, leaf_metric_histogram,
                             metric_heat_map)
    assert metric_heat_map(matrix, ids, values) is not None
    for plot in (leaf_metric_barplot, leaf_metric_histogram):
        ax = plot(1, matrix, ids, values, tolerance=50)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            'within 50 of 100', 'other']


def test_classification_leaf_distances_have_no_tolerance(probability_tree):
    explainer = TreeCounterfactualExplainer(probability_tree)
    with pytest.raises(ValueError):
        explainer.volatility_index(1, tolerance=1)
    assert explainer.support_set_metric(1)[1] == {'same': [3],
                                                  'other': [4]}


def test_class_probability(probability_tree):
    explainer = TreeCounterfactualExplainer(probability_tree)
    x = np.array([5.])  # leaf 1: 'a' (90%)

    def leaves(target):
        return sorted(leaf for found in explainer.explain_instance(
            x, target=target).counterfactuals for leaf in found.per_leaf)
    assert leaves(ClassProbability('b', at_least=0.7)) == [4]
    assert leaves(ClassProbability('b', at_least=0.9)) == []
    # the predicted class with less confidence; the own leaf is never used
    assert leaves(ClassProbability('a', at_least=0.55)) == [3]
    assert leaves(ClassProbability('a', at_least=0.65)) == []
    assert leaves(ClassProbability('a', 'b', at_least=0.6)) == [3, 4]
    assert leaves(ClassProbability('b', at_least=0)) == leaves(OtherClass())
    assert ClassProbability('a', 'b', 'a', at_least=1).labels == ('a', 'b')
    with pytest.raises(ValueError):
        ClassProbability('a', at_least=1.5)
    with pytest.raises(TypeError):
        ClassProbability(at_least=0.5)
    with pytest.raises(TypeError):  # at_least is required
        ClassProbability('a')
    with pytest.raises(ValueError):
        explainer.explain_instance(x, target=ClassProbability(
            'c', at_least=0.5))


@pytest.fixture
def three_classes():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.repeat(['low', 'mid', 'high'], 10)
    return sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)


def test_several_target_classes(three_classes):
    explainer = TreeCounterfactualExplainer(three_classes, ['x'])
    x = np.array([5.])
    both = explainer.explain_instance(x, target=TargetClass('mid', 'high'))
    assert both.target.labels == ('mid', 'high')
    assert both.ranked(with_classes=True) == \
        explainer.explain_instance(x).ranked(with_classes=True)
    assert [classes for _, classes, _ in explainer.explain_instance(
        x, target=TargetClass('high')).ranked(with_classes=True)] == \
        [['high']]
    with pytest.raises(ValueError):  # includes the prediction
        explainer.explain_instance(x, target=TargetClass('low', 'mid'))
    with pytest.raises(ValueError):
        explainer.explain_instance(x, target=TargetClass('mid', 'top'))
    with pytest.raises(TypeError):
        TargetClass()


def test_numerical_columns_described_as_categories():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, (100, 2))
    y = np.where(X[:, 0] > 2.5, 'big', 'small')
    clf = sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)
    names = ['petal_length', 'petal_width']
    with pytest.raises(ValueError, match='numerical'):
        TreeCounterfactualExplainer(clf, names)  # read as feature 'petal'
    from ctreex import FeatureSpec
    explainer = TreeCounterfactualExplainer(
        clf, FeatureSpec.from_column_names(names, numerical=names))
    assert explainer.explain_instance(X[0]).counterfactuals
