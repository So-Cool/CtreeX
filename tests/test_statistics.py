"""Training statistics of the leaves, combinable targets and rankings."""
import json

import numpy as np
import pytest

from ctreex import (AllOf, AnyOf, MinSamples, Not, OtherClass, SameClass,
                    TargetClass, TreeCounterfactualExplainer, TreeStructure,
                    ValueChange, load_tree)

sklearn_tree = pytest.importorskip('sklearn.tree')


@pytest.fixture(scope='module')
def data():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.uniform(0, 100, 400), rng.uniform(0, 10, 400),
                         np.eye(3)[rng.randint(0, 3, 400)]])
    noise = rng.uniform(0, 1, 400) < 0.15
    y = np.where((X[:, 0] > 40) ^ (X[:, 1] > 6) ^ noise, 'yes', 'no')
    return X, y


@pytest.fixture(scope='module')
def clf(data):
    X, y = data
    return sklearn_tree.DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, random_state=0).fit(X, y)


COLUMNS = ['a', 'b', 'job_x', 'job_y', 'job_z']


def test_statistics_are_stored(clf):
    tree = TreeStructure.from_sklearn(clf)
    assert np.array_equal(tree.n_node_samples, clf.tree_.n_node_samples)
    assert np.allclose(tree.weighted_n_node_samples,
                       clf.tree_.weighted_n_node_samples)
    assert np.allclose(tree.impurity, clf.tree_.impurity)
    copy = load_tree(json.loads(tree.to_json()))
    for name in ('n_node_samples', 'weighted_n_node_samples', 'impurity'):
        assert np.allclose(getattr(copy, name), getattr(tree, name))
        assert name not in copy.metadata
    # hand-built trees (and older files) have no statistics
    bare = TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2],
                         [0.5, -2, -2], [[1, 1], [1, 0], [0, 1]], ['n', 'y'])
    assert bare.n_node_samples is None and bare.impurity is None
    assert 'n_node_samples' not in bare.to_dict()
    with pytest.raises(ValueError):
        TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2], [0.5, -2, -2],
                      [[1, 1], [1, 0], [0, 1]], ['n', 'y'],
                      n_node_samples=[2, 1])


def test_context_statistics(clf):
    explainer = TreeCounterfactualExplainer(clf, COLUMNS)
    context = explainer.target_context(explainer.leaf_meta_features
                                       .leaf_ids[0])
    ids = context.leaf_ids
    assert context.leaf_samples.tolist() == \
        clf.tree_.n_node_samples[ids].tolist()
    assert np.allclose(context.leaf_impurity, clf.tree_.impurity[ids])


def _target_leaves(explanation):
    return sorted(leaf for leaves in explanation.support.values()
                  for leaf in leaves['target'])


def test_min_samples(clf, data):
    X, _ = data
    explainer = TreeCounterfactualExplainer(clf, COLUMNS)
    samples = dict(zip(explainer.leaf_meta_features.leaf_ids.tolist(),
                       clf.tree_.n_node_samples[
                           explainer.leaf_meta_features.leaf_ids].tolist()))
    n = int(np.median(list(samples.values())))
    for x in X[:30]:
        e = explainer.explain_instance(x, target=OtherClass() & MinSamples(n))
        default = explainer.explain_instance(x)
        expected = [leaf for leaf in _target_leaves(default)
                    if samples[leaf] >= n]
        assert _target_leaves(e) == expected
        for found in e.counterfactuals:
            assert all(samples[leaf] >= n for leaf in found.per_leaf)
    for invalid in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            MinSamples(invalid)
    bare = TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2],
                         [0.5, -2, -2], [[1, 1], [1, 0], [0, 1]], ['n', 'y'])
    with pytest.raises(ValueError):  # no statistics
        TreeCounterfactualExplainer(bare, ['x']).explain_instance(
            np.array([0.]), target=OtherClass() & MinSamples(1))


def test_combined_targets(clf, data):
    X, _ = data
    explainer = TreeCounterfactualExplainer(clf, COLUMNS)
    x = X[0]
    context = explainer.target_context(explainer.predict_leaf(x))
    other = OtherClass().select(context)
    many = MinSamples(30).select(context)
    assert (OtherClass() & MinSamples(30)).select(context).tolist() == \
        (other & many).tolist()
    assert (OtherClass() | MinSamples(30)).select(context).tolist() == \
        (other | many).tolist()
    assert (~SameClass()).select(context).tolist() == other.tolist()
    assert AllOf(OtherClass(), MinSamples(30), Not(MinSamples(60))) \
        .select(context).tolist() == (
            other & many & ~MinSamples(60).select(context)).tolist()
    assert AnyOf(SameClass()) == AnyOf(SameClass())
    assert _target_leaves(explainer.explain_instance(
        x, target=~SameClass())) == _target_leaves(
            explainer.explain_instance(x))
    # the parts are validated
    prediction = context.prediction
    with pytest.raises(ValueError):
        explainer.explain_instance(
            x, target=TargetClass(prediction) & MinSamples(1))
    with pytest.raises(ValueError):  # a regression target
        explainer.explain_instance(x, target=OtherClass() | ValueChange(1))
    with pytest.raises(TypeError):
        AllOf()
    with pytest.raises(TypeError):
        OtherClass() & 'yes'
    with pytest.raises(TypeError):
        Not('yes')


def test_rank_by(clf, data):
    X, _ = data
    explainer = TreeCounterfactualExplainer(clf, COLUMNS)
    ids = explainer.leaf_meta_features.leaf_ids.tolist()
    samples = dict(zip(ids, clf.tree_.n_node_samples[ids].tolist()))
    value = clf.tree_.value[ids, 0, :]
    purity = dict(zip(ids, (value.max(axis=1) / value.sum(axis=1)).tolist()))
    for x in X[:30]:
        default = explainer.explain_instance(x)
        assert explainer.explain_instance(
            x, rank_by='frequency').ranked() == default.ranked()
        for rank_by, score, aggregate in (('samples', samples, sum),
                                          ('purity', purity, min)):
            e = explainer.explain_instance(x, rank_by=rank_by)
            for found in e.counterfactuals:
                # the same counterfactuals, possibly in another order
                reference = default.at_distance(found.distance)
                assert sorted(map(repr, found.ranked.values())) == \
                    sorted(map(repr, reference.ranked.values()))
                if found.distance > 1:
                    leaves = [_source(found, rank)
                              for rank in sorted(found.ranked)]
                    values = [score[leaf] for leaf in leaves]
                    assert values == sorted(values, reverse=True)
    with pytest.raises(ValueError):
        explainer.explain_instance(X[0], rank_by='size')


def _source(found, rank):
    """The leaf a ranked counterfactual (distance > 1) comes from."""
    leaves = [leaf for leaf, changes in found.per_leaf.items()
              if changes == found.ranked[rank]]
    return leaves[0]


def _two_ways(samples, purity):
    """x <= 1 and z <= 0.5 -> 'n' (leaf 2); x <= 1 and z > 0.5 -> 'y'
    (leaf 3); x > 1 -> 'y' (leaf 4); ``samples`` and ``purity`` of the 'y'
    leaves 3 and 4."""
    value = [[1, 1], [1, 1], [10, 0]]
    for n, p in zip(samples, purity):
        value.append([n * (1 - p), n * p])
    return TreeStructure(
        children_left=[1, 2, -1, -1, -1], children_right=[4, 3, -1, -1, -1],
        feature=[0, 1, -2, -2, -2], threshold=[1., 0.5, -2, -2, -2],
        value=value, classes=['n', 'y'], feature_names=['x', 'z'],
        n_node_samples=[10 + sum(samples), 10 + samples[0], 10] +
        list(samples))


def test_rank_by_at_distance_one():
    x = np.array([0., 0.])

    def first(tree, rank_by):
        found = TreeCounterfactualExplainer(tree).explain_instance(
            x, rank_by=rank_by).at_distance(1)
        assert len(found.ranked) == 2
        return list(found.ranked[0])
    # leaf 3 (change z): 5 samples, 60%; leaf 4 (change x): 50, 98%
    tree = _two_ways([5, 50], [0.6, 0.98])
    assert first(tree, 'samples') == ['x']
    assert first(tree, 'purity') == ['x']
    # leaf 3 (change z): 80 samples, 70%; leaf 4 (change x): 50, 98%
    tree = _two_ways([80, 50], [0.7, 0.98])
    assert first(tree, 'samples') == ['z']
    assert first(tree, 'purity') == ['x']


def test_rank_by_regression():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.where(X[:, 0] < 20, X[:, 0], 100 + (X[:, 0] % 3))
    reg = sklearn_tree.DecisionTreeRegressor(max_depth=3,
                                             random_state=0).fit(X, y)
    explainer = TreeCounterfactualExplainer(reg, ['x'])
    e = explainer.explain_instance(np.array([0.]), target=ValueChange(50),
                                   rank_by='purity')
    ids = explainer.leaf_meta_features.leaf_ids.tolist()
    impurity = dict(zip(ids, reg.tree_.impurity[ids].tolist()))
    found = e.at_distance(1)
    order = [min(impurity[leaf] for leaf in found.per_leaf
                 if found.leaf_values[leaf] == low)
             for low, _ in (found.ranked_values[rank]
                            for rank in sorted(found.ranked))]
    assert order == sorted(order)  # lower impurity first
    bare = TreeStructure.from_dict({
        k: v for k, v in TreeStructure.from_sklearn(reg).to_dict().items()
        if k != 'impurity'})
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(bare, ['x']).explain_instance(
            np.array([0.]), target=ValueChange(50), rank_by='purity')
