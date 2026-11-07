import numpy as np
import pytest

from ctreex import TreeCounterfactualExplainer, TreeStructure
from ctreex.counterfactuals import rank_counterfactuals
from conftest import load_case


def test_deterministic_order():
    # duplicates are removed keeping the order in which they were found
    per_leaf = {1: {'job': {'is': ['a'], 'wannabe': ['c', 'b']}},
                2: {'job': {'is': ['a'], 'wannabe': ['b', 'd']}}}
    per_feature = {'job': [1, 2]}
    ranked = rank_counterfactuals(per_leaf, per_feature, 1)
    assert ranked[0]['job']['wannabe'] == ['c', 'b', 'd']


def test_no_missing_distances():
    # leaves lie at every distance between 0 and the largest one
    case = load_case('single_threshold_case.json')
    explainer = TreeCounterfactualExplainer(case['tree'])
    for x in np.array(case['X']):
        e = explainer.explain_instance(x)
        assert sorted(e.support) == list(range(max(e.support) + 1))
    # distances beyond the largest one are skipped
    e = explainer.explain_instance(np.array(case['X'])[0], max_distance=50)
    assert e.distances == sorted(d for d in e.support if d > 0)


def test_same_class_only_distance():
    tree = TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 1, -2, -2], threshold=[0.5, -2, 0.5, -2, -2],
        value=[[2, 2], [2, 0], [0, 2], [1, 0], [0, 2]], classes=['n', 'y'],
        feature_names=['x0', 'x1'])
    x = np.array([0., 1.])  # leaf 1 ('n'); leaf 4 ('y') needs x0 to change
    explainer = TreeCounterfactualExplainer(tree)
    e = explainer.explain_instance(x)
    assert e.prediction == 'n'
    # leaf 3 (distance 2) has the same class -- no counterfactuals there
    assert e.distances == [1, 2]
    assert e.at_distance(2).per_leaf == {}
    assert e.at_distance(1).ranked[0] == {
        'x0': {'is': [0], 'wannabe': [['0.5', None]]}}
    assert explainer.describe(
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


def test_complete_numerical_intervals():
    case = load_case('single_threshold_case.json')
    explainer = TreeCounterfactualExplainer(case['tree'])
    names = [m.name for m in explainer.meta_features]
    # features split at a single threshold have both intervals
    for feature in case['single_threshold']:
        lower = [n for n in names if n.startswith(feature + '<=')]
        assert len(lower) == 1
        upper = '{}<{}'.format(lower[0].split('<=')[1], feature)
        assert upper in names


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
    explainer = TreeCounterfactualExplainer(case['tree'])
    excluded = restriction.get('excluded', [])
    included = restriction.get('included', [])

    def allowed(feature):
        return feature not in excluded and (not included
                                            or feature in included)
    for x in np.array(case['X']):
        full = explainer.explain_instance(x)
        restricted = explainer.explain_instance(x, **restriction)
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


def _apply(spec, values, changes):
    """Changes a data point as a (single-leaf) counterfactual says."""
    new = dict(values)
    for feature, change in changes.items():
        wannabe = change['wannabe'][0]
        if spec.kind(feature) == 'numerical':
            lower, upper = wannabe
            new[feature] = (float(upper) if upper is not None
                            else float(lower) + 1)
        else:
            new[feature] = wannabe
    return spec.encode(new)


def _category_gap_tree():
    # age <= 40 -> (job_b <= 0.5 -> leaf 2 'no', else leaf 3 'yes');
    # age > 40  -> (job_a <= 0.5 -> leaf 5 'yes', else leaf 6 'no');
    # the tree never splits on job_c
    return TreeStructure(
        children_left=[1, 2, -1, -1, 5, -1, -1],
        children_right=[4, 3, -1, -1, 6, -1, -1],
        feature=[0, 2, -2, -2, 1, -2, -2],
        threshold=[40., 0.5, -2, -2, 0.5, -2, -2],
        value=[[4, 4], [2, 2], [2, 0], [0, 2], [2, 2], [0, 2], [2, 0]],
        classes=['no', 'yes'], n_features=4,
        feature_names=['age', 'job_a', 'job_b', 'job_c'])


def test_complete_categorical_features():
    tree = _category_gap_tree()
    x = np.array([30., 0, 0, 1])  # age 30, job c -> leaf 2 ('no')
    explainer = TreeCounterfactualExplainer(tree)
    # job c is described although the tree never splits on it
    assert [m.name for m in explainer.meta_features
            if m.feature == 'job'] == ['job_a', 'job_b', 'job_c']
    e = explainer.explain_instance(x)
    assert e.leaf == 2
    assert e.leaf_distances.tolist() == [0, 1, 1, 2]
    # "age more than 40" alone reaches leaf 5 ('yes')
    assert e.at_distance(1).per_leaf[5] == {
        'age': {'is': [30], 'wannabe': [['40.0', None]]}}


def test_counterfactuals_reach_their_leaves():
    case = load_case('single_threshold_case.json')
    explainer = TreeCounterfactualExplainer(case['tree'])
    spec = explainer.spec
    ids = explainer.leaf_meta_features.leaf_ids.tolist()
    for x in np.array(case['X']):
        e = explainer.explain_instance(x)
        assert e.leaf_distances[ids.index(e.leaf)] == 0
        values = spec.decode(x)
        for found in e.counterfactuals:
            for leaf, changes in found.per_leaf.items():
                assert len(changes) == found.distance
                new = _apply(spec, values, changes)
                assert explainer.predict_leaf(new) == leaf


def test_feature_names_check(toy_tree, toy_spec):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        TreeCounterfactualExplainer(toy_tree, toy_spec)  # same names
    with pytest.warns(UserWarning):
        TreeCounterfactualExplainer(toy_tree, ['age', 'job_x', 'job_y',
                                               'job_z'])


@pytest.fixture
def three_classes():
    sklearn_tree = pytest.importorskip('sklearn.tree')
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.repeat(['low', 'mid', 'high'], 10)
    return sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)


def test_multi_class(three_classes):
    x = np.array([5.])
    fixed = TreeCounterfactualExplainer(three_classes, ['x'])
    e = fixed.explain_instance(x)
    assert e.prediction == 'low'
    found = e.at_distance(1)
    assert sorted(found.leaf_classes.values()) == ['high', 'mid']
    # one counterfactual per class
    described = {tuple(classes): fixed.describe({0: cf})[0]['x']['wannabe']
                 for _, classes, cf in e.ranked(with_classes=True)}
    assert described == {('mid',): ['between 9.5 and 19.5'],
                         ('high',): ['more than 19.5']}


def test_target_class(three_classes):
    explainer = TreeCounterfactualExplainer(three_classes, ['x'])
    x = np.array([5.])
    e = explainer.explain_instance(x, target_class='high')
    assert [(d, classes) for d, classes, _ in e.ranked(with_classes=True)] \
        == [(1, ['high'])]
    assert explainer.describe(e.counterfactuals[0].ranked)[0]['x'][
        'wannabe'] == ['more than 19.5']
    with pytest.raises(ValueError):
        explainer.explain_instance(x, target_class='low')  # the prediction
    with pytest.raises(ValueError):
        explainer.explain_instance(x, target_class='very high')


def test_ranked_counterfactuals_lead_to_one_class():
    case = load_case('single_threshold_case.json')
    explainer = TreeCounterfactualExplainer(case['tree'])
    for x in np.array(case['X']):
        e = explainer.explain_instance(x)
        assert all(len(classes) == 1
                   for _, classes, _ in e.ranked(with_classes=True))
