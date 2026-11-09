"""Instance-contrastive explanations."""
import numpy as np
import pytest

from ctreex import Leaves, TargetClass, TreeCounterfactualExplainer

from test_supportive import COLUMNS  # noqa: E402

sklearn_tree = pytest.importorskip('sklearn.tree')


@pytest.fixture(scope='module')
def data():
    rng = np.random.RandomState(3)
    X = np.column_stack([rng.uniform(0, 10, (600, 3)),
                         np.eye(3)[rng.randint(0, 3, 600)]])
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] - 2 * X[:, 5] + \
        rng.normal(0, 1, 600)
    return X, score


@pytest.fixture(scope='module')
def classifier(data):
    X, score = data
    y = np.where(score > 3, 'high', np.where(score > -1, 'mid', 'low'))
    clf = sklearn_tree.DecisionTreeClassifier(
        max_leaf_nodes=14, random_state=0).fit(X, y)
    return TreeCounterfactualExplainer(clf, COLUMNS), X


def _contains(change, reference):
    """Whether the new values include the value of the reference."""
    if isinstance(change[0], list):
        return any((lo is None or float(lo) < reference) and
                   (hi is None or reference <= float(hi))
                   for lo, hi in change)
    return reference in change


def test_explain_contrast(classifier):
    explainer, X = classifier
    predictions = explainer.tree.predict(X)
    checked = 0
    for i in range(30):
        x = X[i]
        for j in range(30, 60):
            reference = X[j]
            if predictions[i] == predictions[j]:
                continue
            e = explainer.explain_contrast(x, reference)
            assert e.reference == explainer.spec.decode(reference)
            assert e.target == TargetClass(predictions[j])
            plain = explainer.explain_instance(
                x, target=TargetClass(predictions[j]))
            values = explainer.spec.decode(reference)
            for found in e.counterfactuals:
                reference_found = plain.at_distance(found.distance)
                # the same counterfactuals, reordered
                assert sorted(map(repr, found.ranked.values())) == sorted(
                    map(repr, reference_found.ranked.values()))
                assert found.per_leaf == reference_found.per_leaf
                counts = []
                for rank, cf in found.ranked.items():
                    matching = [f for f, change in cf.items()
                                if _contains(change['wannabe'], values[f])]
                    assert found.reference_matches[rank] == matching
                    assert found.ranked_classes[rank] == [predictions[j]]
                    counts.append(len(matching))
                assert counts == sorted(counts, reverse=True)
            checked += 1
            break
    assert checked > 10


def test_same_leaf(classifier):
    explainer, X = classifier
    leaves = explainer.tree.apply(X)
    predictions = explainer.tree.predict(X)
    pairs = [(i, j) for i in range(40) for j in range(40)
             if predictions[i] == predictions[j] and leaves[i] != leaves[j]]
    assert pairs
    i, j = pairs[0]
    with pytest.raises(ValueError):  # the same prediction
        explainer.explain_contrast(X[i], X[j])
    e = explainer.explain_contrast(X[i], X[j], same_leaf=True)
    assert e.target == Leaves(int(leaves[j]))
    assert {leaf for found in e.counterfactuals
            for leaf in found.per_leaf} == {int(leaves[j])}
    same = [k for k in range(40) if leaves[k] == leaves[i] and k != i]
    if same:
        with pytest.raises(ValueError):
            explainer.explain_contrast(X[i], X[same[0]], same_leaf=True)
    with pytest.raises(ValueError):  # no tolerance for classifiers
        explainer.explain_contrast(X[i], X[j], same_leaf=True, tolerance=1)


def test_contrast_options(classifier):
    explainer, X = classifier
    predictions = explainer.tree.predict(X)
    j = next(k for k in range(1, 100) if predictions[k] != predictions[0])
    restricted = explainer.explain_contrast(X[0], X[j], excluded=['a'])
    assert all('a' not in cf for _, cf in restricted.ranked())
    with pytest.raises(TypeError):
        explainer.explain_contrast(X[0], X[j], target=TargetClass('mid'))
    with pytest.raises(ValueError):
        explainer.explain_contrast(X[0], X[j], tolerance=1)


def test_regression_contrast(data):
    X, score = data
    reg = sklearn_tree.DecisionTreeRegressor(
        max_leaf_nodes=14, random_state=0).fit(X, 10 * score)
    explainer = TreeCounterfactualExplainer(reg, COLUMNS)
    predictions = reg.predict(X)
    i, j = int(np.argmin(predictions)), int(np.argmax(predictions))
    with pytest.raises(ValueError):  # a tolerance is required
        explainer.explain_contrast(X[i], X[j])
    e = explainer.explain_contrast(X[i], X[j], tolerance=5)
    assert e.counterfactuals
    for found in e.counterfactuals:
        for value in found.leaf_values.values():
            assert abs(value - predictions[j]) <= 5
    with pytest.raises(ValueError):  # within the tolerance
        explainer.explain_contrast(X[i], X[j], tolerance=1e6)
    assert explainer.explain_contrast(X[i], X[j], same_leaf=True).target == \
        Leaves(int(reg.apply(X[j:j + 1])[0]))


def test_leaves_target(classifier):
    explainer, X = classifier
    leaf = int(explainer.leaf_meta_features.leaf_ids[0])
    assert Leaves(leaf, leaf).ids == (leaf,)
    with pytest.raises(TypeError):
        Leaves()
    with pytest.raises(ValueError):  # the root is not a leaf
        explainer.explain_instance(X[0], target=Leaves(0))
