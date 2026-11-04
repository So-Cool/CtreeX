import dataclasses

import numpy as np
import pytest

from ctreex import Fixes, TreeCounterfactualExplainer, TreeStructure
from ctreex.counterfactuals import rank_counterfactuals
from ctreex.fixes import resolve_fixes
from conftest import load_case


def test_resolve_fixes():
    assert resolve_fixes('legacy') == Fixes.legacy()
    assert resolve_fixes('fixed') == Fixes.fixed()
    custom = resolve_fixes('legacy', {'deterministic_order': True})
    assert custom.deterministic_order and \
        not custom.continue_past_missing_distance
    with pytest.raises(ValueError):
        resolve_fixes('newest')
    with pytest.raises(TypeError):
        resolve_fixes('fixed', ['deterministic_order'])


def test_deterministic_order():
    per_leaf = {1: {'job': {'is': ['a'], 'wannabe': ['c', 'b']}},
                2: {'job': {'is': ['a'], 'wannabe': ['b', 'd']}}}
    per_feature = {'job': [1, 2]}
    ranked = rank_counterfactuals(per_leaf, per_feature, 1,
                                  deterministic_order=True)
    assert ranked[0]['job']['wannabe'] == ['c', 'b', 'd']


def test_continue_past_missing_distance():
    case = load_case('distance_gap_case.json')
    x = np.array(case['x'])
    legacy = TreeCounterfactualExplainer(case['tree'],
                                           implementation='legacy')
    fixed = TreeCounterfactualExplainer(case['tree'], implementation='fixed')
    e_legacy = legacy.explain_instance(x)
    e_fixed = fixed.explain_instance(x)
    assert sorted(e_legacy.support) == [0, 1, 3]  # no leaf at distance 2
    assert e_legacy.distances == [1]  # the legacy search stops at distance 2
    assert e_fixed.distances == [1, 3]
    assert len(e_fixed.at_distance(3).per_leaf) == 1
    only_this_fix = TreeCounterfactualExplainer(
        case['tree'], implementation='legacy',
        fixes={'continue_past_missing_distance': True})
    assert only_this_fix.explain_instance(x).distances == [1, 3]


def test_same_class_only_distance():
    tree = TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 1, -2, -2], threshold=[0.5, -2, 0.5, -2, -2],
        value=[[2, 2], [2, 0], [0, 2], [1, 0], [0, 2]], classes=['n', 'y'],
        feature_names=['x0', 'x1'])
    x = np.array([0., 1.])  # leaf 1 ('n'); leaf 4 ('y') needs x0 to change
    for implementation in ('legacy', 'fixed'):
        e = TreeCounterfactualExplainer(tree, implementation=implementation)\
            .explain_instance(x)
        assert e.prediction == 'n'
        # leaf 3 (distance 2) has the same class -- no counterfactuals there
        assert e.distances == [1, 2]
        assert e.at_distance(2).per_leaf == {}
        # x0 is split at a single threshold: the legacy implementation has
        # no "0.5<x0" meta-feature and describes the interval with a tuple
        interval = ('0.5', None) if implementation == 'legacy' \
            else ['0.5', None]
        assert e.at_distance(1).ranked[0] == {
            'x0': {'is': [0], 'wannabe': [interval]}}
        assert TreeCounterfactualExplainer(tree).describe(
            e.at_distance(1).ranked)[0]['x0']['wannabe'] == ['more than 0.5']


def test_explainer_on_sklearn_model():
    sklearn = pytest.importorskip('sklearn.tree')
    rng = np.random.RandomState(1)
    X = np.column_stack([rng.randint(18, 70, 300),
                         np.eye(2)[rng.randint(0, 2, 300)]])
    y = np.where((X[:, 0] > 40) | (X[:, 2] == 1), 1, 0)
    clf = sklearn.DecisionTreeClassifier(random_state=0).fit(X, y)
    explainer = TreeCounterfactualExplainer(clf, ['age', 'owner_no',
                                                  'owner_yes'])
    e = explainer.explain_instance(X[0])
    assert e.prediction == clf.predict(X[:1])[0]
    for distance, cf in e.ranked():
        assert len(cf) == distance or distance > 1
    described = explainer.describe(e.counterfactuals[0].ranked)
    assert all(isinstance(k, int) for k in described)


def test_input_validation(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    with pytest.raises(ValueError):
        explainer.explain_instance(np.zeros(3))
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(toy_tree, ['a', 'b'])
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(toy_tree, toy_spec,
                                    class_names=['only one'])
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(toy_tree, toy_spec,
                                    implementation='original')


def test_complete_numerical_intervals():
    case = load_case('single_threshold_case.json')
    base = dataclasses.replace(Fixes.legacy(), deterministic_order=True)
    legacy = TreeCounterfactualExplainer(case['tree'], fixes=base)
    complete = TreeCounterfactualExplainer(
        case['tree'],
        fixes=dataclasses.replace(base, complete_numerical_intervals=True))
    legacy_names = [m.name for m in legacy.meta_features]
    complete_names = [m.name for m in complete.meta_features]
    for feature in case['single_threshold']:
        lower = [n for n in legacy_names if n.startswith(feature + '<=')]
        assert len(lower) == 1
        upper = '{}<{}'.format(lower[0].split('<=')[1], feature)
        assert upper not in legacy_names and upper in complete_names
    assert len(complete_names) == len(legacy_names) + len(
        case['single_threshold'])
    # the counterfactuals of data points and their distances do not change
    for x in np.array(case['X']):
        a, b = legacy.explain_instance(x), complete.explain_instance(x)
        assert np.array_equal(a.leaf_distances, b.leaf_distances)
        assert a.support == b.support
        assert [(c.distance, legacy.describe(c.ranked))
                for c in a.counterfactuals] == \
            [(c.distance, complete.describe(c.ranked))
             for c in b.counterfactuals]
    # ... but the leaf-to-leaf distances do
    assert not np.array_equal(legacy.leaf_distance_matrix()[0],
                              complete.leaf_distance_matrix()[0])


def test_disjoint_regions():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    X = np.arange(31, dtype=float).reshape(-1, 1)
    y = np.where((X[:, 0] <= 10) | (X[:, 0] > 20), 'yes', 'no')
    clf = sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)
    explainer = TreeCounterfactualExplainer(clf, ['x'])
    explanation = explainer.explain_instance(np.array([15.]))
    assert explanation.prediction == 'no'
    assert explainer.describe(explanation.at_distance(1).ranked) == {
        0: {'x': {'is': [15], 'wannabe': ['less or equal to 10.5',
                                          'more than 20.5']}}}


RESTRICTIONS = [{'excluded': ['a']}, {'excluded': ['d', 'c']},
                {'included': ['b', 'd']}, {'included': ['a']},
                {'excluded': ['b'], 'included': ['a', 'b', 'c']}]


@pytest.mark.parametrize('restriction', RESTRICTIONS)
def test_whole_counterfactual_restrictions(restriction):
    case = load_case('single_threshold_case.json')
    fixed = TreeCounterfactualExplainer(case['tree'])
    legacy = TreeCounterfactualExplainer(case['tree'],
                                         implementation='legacy')
    excluded = restriction.get('excluded', [])
    included = restriction.get('included', [])

    def allowed(feature):
        return feature not in excluded and (not included
                                            or feature in included)
    for x in np.array(case['X']):
        full = fixed.explain_instance(x)
        restricted = fixed.explain_instance(x, **restriction)
        for found in restricted.counterfactuals:
            reference = full.at_distance(found.distance)
            for leaf, changes in found.per_leaf.items():
                # complete counterfactuals of allowed features only
                assert len(changes) == found.distance
                assert all(allowed(f) for f in changes)
                assert changes == reference.per_leaf[leaf]
            # leaves are skipped only when they need a forbidden feature
            for leaf, changes in reference.per_leaf.items():
                if all(allowed(f) for f in changes):
                    assert leaf in found.per_leaf
            assert all(cf for cf in found.ranked.values())
        # the legacy implementation truncates the counterfactuals instead
        legacy_full = legacy.explain_instance(x)
        legacy_restricted = legacy.explain_instance(x, **restriction)
        for found in legacy_restricted.counterfactuals:
            reference = legacy_full.at_distance(found.distance)
            assert set(found.per_leaf) == set(reference.per_leaf)
            for leaf, changes in found.per_leaf.items():
                assert changes == {f: v for f, v in
                                   reference.per_leaf[leaf].items()
                                   if allowed(f)}


def test_restriction_fix_switch():
    case = load_case('single_threshold_case.json')
    truncating = TreeCounterfactualExplainer(
        case['tree'], fixes={'whole_counterfactual_restrictions': False})
    found_truncated = found_incomplete = False
    for restriction in RESTRICTIONS:
        for x in np.array(case['X']):
            for found in truncating.explain_instance(
                    x, **restriction).counterfactuals:
                for changes in found.per_leaf.values():
                    found_incomplete |= len(changes) < found.distance
                    found_truncated |= not changes
    assert found_incomplete and found_truncated
