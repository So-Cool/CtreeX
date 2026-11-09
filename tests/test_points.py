"""Concrete data points, random samples and exemplars of counterfactuals."""
import numpy as np
import pytest

from ctreex import (Feature, FeatureSpec, SameClass,
                    TreeCounterfactualExplainer, ValueChange)
from ctreex import points

sklearn_tree = pytest.importorskip('sklearn.tree')

COLUMNS = ['a', 'n', 'job_x', 'job_y', 'job_z']
above, at_most = points.above, points.at_most


def _data(seed=4, n=600):
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.uniform(-5, 5, n), rng.randint(0, 40, n),
                         np.eye(3)[rng.randint(0, 3, n)]])
    score = X[:, 0] + X[:, 1] / 8 + 2 * X[:, 3] + rng.normal(0, 1, n)
    return X, score


def _spec(X):
    return FeatureSpec.from_column_names(COLUMNS, features=[
        Feature('n', 'numerical', integer=True),
        Feature('job', 'categorical', ordered=True)]).fit_ranges(X)


@pytest.fixture(scope='module', params=['classifier', 'regressor'])
def model(request):
    X, score = _data()
    if request.param == 'classifier':
        estimator = sklearn_tree.DecisionTreeClassifier(
            max_leaf_nodes=16, random_state=0).fit(
                X, np.where(score > 4, 'yes', 'no'))
        target = None
    else:
        estimator = sklearn_tree.DecisionTreeRegressor(
            max_leaf_nodes=16, random_state=0).fit(X, 10 * score)
        target = ValueChange(+10)
    return estimator, TreeCounterfactualExplainer(estimator, _spec(X)), \
        target, X


def test_float32_bounds():
    for t in (0.1, 2.5, -3.7, 1e-8, 22.5):
        assert above(t) > t and np.float32(above(t)) == above(t)
        assert at_most(t) <= t and np.float32(at_most(t)) == at_most(t)
        assert np.nextafter(np.float32(above(t)), np.float32(-np.inf)) <= t


def test_points_reach_the_counterfactual(model):
    estimator, explainer, target, X = model
    checked = 0
    for x in X[:25]:
        explanation = explainer.explain_instance(x, target=target)
        for distance, cf in explanation.ranked():
            leaves = explainer.counterfactual_leaves(x, cf)
            rows = [explainer.counterfactual_point(x, cf),
                    explainer.counterfactual_point(x, cf, how='midpoint')]
            rows += list(explainer.sample_counterfactual(
                x, cf, n=5, random_state=checked))
            rows = np.array(rows)
            # scikit-learn (32-bit data) and TreeStructure agree
            assert estimator.apply(rows).tolist() == \
                explainer.tree.apply(rows).tolist()
            assert set(explainer.tree.apply(rows).tolist()) <= set(leaves)
            for row in rows:
                values = explainer.decode(row)
                original = explainer.decode(x)
                assert {f for f in values if values[f] != original[f]} <= \
                    set(cf)
                assert float(values['n']).is_integer()
            checked += 1
    assert checked > 20


def test_nearest_points(model):
    _, explainer, target, X = model
    for x in X[:25]:
        explanation = explainer.explain_instance(x, target=target)
        original = explainer.decode(x)
        for _, cf in explanation.ranked():
            near = explainer.decode(explainer.counterfactual_point(x, cf))
            if 'a' in cf:
                # no 32-bit number between the data point and the nearest
                # value is in the range
                value, a = near['a'], original['a']
                step = np.nextafter(np.float32(value), np.float32(
                    -np.inf if value > a else np.inf))
                assert not _inside(float(step), cf['a']['wannabe'])
            if 'n' in cf:
                value, n = near['n'], original['n']
                assert not _inside(value - np.sign(value - n),
                                   cf['n']['wannabe'])


def _inside(value, wannabe):
    return any((lo is None or value > float(lo)) and
               (hi is None or value <= float(hi)) for lo, hi in wannabe)


def test_contextualised_counterfactuals(model):
    _, explainer, target, X = model
    x = X[0]
    for c in explainer.contextualise(explainer.explain_instance(
            x, target=target)):
        rows = explainer.sample_counterfactual(x, c, n=3, random_state=1)
        assert rows.shape == (3, len(COLUMNS))
        explainer.counterfactual_point(x, c)


def test_sampling_is_reproducible(model):
    _, explainer, target, X = model
    x = X[1]
    cf = explainer.explain_instance(x, target=target).ranked()[0][1]
    a = explainer.sample_counterfactual(x, cf, n=4, random_state=7)
    b = explainer.sample_counterfactual(x, cf, n=4,
                                        random_state=np.random.default_rng(7))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        explainer.sample_counterfactual(x, cf, n=0)
    with pytest.raises(TypeError):
        explainer.sample_counterfactual(x, [cf])


def test_value_helpers():
    real = Feature('a', 'numerical', minimum=-10., maximum=10.)
    whole = Feature('n', 'numerical', integer=True, minimum=0, maximum=40)
    regions = [[None, '-2.5'], ['3.5', None]]
    assert points.nearest_value(real, regions, 1.) == above(3.5)
    assert points.nearest_value(real, regions, -1.) == at_most(-2.5)
    assert points.nearest_value(whole, [['4.5', '9.5']], 2) == 5
    assert points.nearest_value(whole, [['4.5', '9.5']], 12) == 9
    stepped = Feature('p', 'numerical', step=0.01)
    assert points.nearest_value(stepped, [['3.5', None]], 0.) == \
        pytest.approx(3.51)
    assert points.midpoint_value(real, [['3.5', None]], 0.) == \
        pytest.approx(6.75)
    assert points.midpoint_value(whole, [['4.5', '9.5']], 2) == 7
    with pytest.raises(ValueError):  # unbounded without a maximum
        points.midpoint_value(Feature('b', 'numerical'), [['3.5', None]], 0.)
    with pytest.raises(ValueError):  # no whole number in the range
        points.nearest_value(whole, [['4.2', '4.7']], 2)
    rng = np.random.default_rng(0)
    values = [points.sample_value(real, regions, rng) for _ in range(2000)]
    assert all(_inside(v, regions) and -10 <= v <= 10 for v in values)
    # the longer range (3.5, 10] is drawn more often than [-10, -2.5]
    assert 0.4 < np.mean([v < 0 for v in values]) < 0.6
    integers = [points.sample_value(whole, [['4.5', '9.5']], rng)
                for _ in range(200)]
    assert set(integers) == {5, 6, 7, 8, 9}
    with pytest.raises(ValueError):  # outside of the bounds
        points.sample_value(whole, [['50.5', None]], rng)
    sizes = Feature('size', 'categorical', values=['s', 'm', 'l'],
                    ordered=True)
    assert points.nearest_value(sizes, ['s', 'l'], 'm') == 's'
    assert points.nearest_value(sizes, ['l', 's'], 's') == 's'
    assert points.sample_value(sizes, ['l'], rng) == 'l'


def test_features_with_ranges():
    X, _ = _data()
    spec = _spec(X)
    assert spec.features['a'].minimum == X[:, 0].min()
    assert spec.features['n'].maximum == X[:, 1].max()
    with pytest.raises(ValueError):
        Feature('job', 'categorical', minimum=0)
    with pytest.raises(ValueError):
        Feature('a', 'numerical', minimum=2, maximum=1)
    with pytest.raises(ValueError):
        Feature('a', 'numerical', step=0)


def test_hidden_thresholds(model):
    _, explainer, target, X = model
    x = X[2]
    explanation = explainer.explain_instance(x, target=target)
    ranked = dict(enumerate(cf for _, cf in explanation.ranked()))
    hidden = explainer.describe(ranked, hide_thresholds=True, random_state=3)
    assert hidden == explainer.describe(ranked, hide_thresholds=True,
                                        random_state=3)
    described = explainer.describe(ranked)
    for key, cf in ranked.items():
        for name, part in cf.items():
            display = explainer.spec.features[name].display_name
            if explainer.spec.features[name].kind == 'numerical':
                value, = hidden[key][display]['wannabe']
                assert _inside(value, part['wannabe'])
            else:
                assert hidden[key][display] == described[key][display]


def test_exemplars(model):
    _, explainer, target, X = model
    x = X[3]
    cf = explainer.explain_instance(x, target=target).ranked()[0][1]
    leaves = explainer.counterfactual_leaves(x, cf)
    found = explainer.exemplars(X, leaves)
    assert set(explainer.tree.apply(found).tolist()) <= set(leaves)
    assert len(found) == np.isin(explainer.tree.apply(X), leaves).sum()
    few = explainer.exemplars(X, leaves, n=3, random_state=0)
    assert len(few) == min(3, len(found))
    assert np.array_equal(few, explainer.exemplars(X, leaves, n=3,
                                                   random_state=0))
    assert len(explainer.exemplars(X, leaves[0])) > 0
    with pytest.raises(ValueError):
        explainer.exemplars(X, [0])  # the root is not a leaf


def test_decode(model):
    _, explainer, _, X = model
    assert explainer.decode(X[0]) == explainer.spec.decode(X[0])
    assert explainer.decode(X[:2]) == [explainer.spec.decode(X[0]),
                                       explainer.spec.decode(X[1])]


def test_same_class_counterfactuals(model):
    estimator, explainer, target, X = model
    if target is not None:
        return
    x = X[5]
    for _, cf in explainer.explain_instance(x, target=SameClass()).ranked():
        rows = explainer.sample_counterfactual(x, cf, n=3, random_state=2)
        assert set(estimator.predict(rows)) == set(
            estimator.predict(x.reshape(1, -1)))
