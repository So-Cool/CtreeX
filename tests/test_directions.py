"""Ordered features, direction constraints and monotonicity audits."""
import numpy as np
import pytest

from ctreex import (Feature, FeatureSpec, TreeCounterfactualExplainer,
                    TreeStructure)
from ctreex.counterfactuals import in_direction
from ctreex.meta_features import MetaFeature

from test_supportive import COLUMNS, Grid  # noqa: E402

sklearn_tree = pytest.importorskip('sklearn.tree')


def test_ordered_features():
    columns = [('size', 'm'), ('size', 's'), ('size', 'l'), ('x', None)]
    spec = FeatureSpec(columns, features=[
        Feature('size', 'categorical', values=['s', 'm', 'l'], ordered=True)])
    assert spec.features['size'].values == ['s', 'm', 'l']
    assert spec.features['size'].ordered
    # without values the order of the columns is used
    spec = FeatureSpec(columns, features=[
        Feature('size', 'categorical', ordered=True)])
    assert spec.features['size'].values == ['m', 's', 'l']
    # unordered features list their categories in the order of the columns
    spec = FeatureSpec(columns, features=[
        Feature('size', 'categorical', values=['s', 'm', 'l'])])
    assert spec.features['size'].values == ['m', 's', 'l']
    with pytest.raises(ValueError):
        FeatureSpec(columns, features=[Feature(
            'size', 'categorical', values=['s', 'm', 'xl'], ordered=True)])
    with pytest.raises(ValueError):
        Feature('x', 'numerical', ordered=True)


def test_in_direction():
    numbers = [MetaFeature('x<=1', 'x', 'numerical', 0, upper='1.0'),
               MetaFeature('1<x<=5', 'x', 'numerical', 0, lower='1.0',
                           upper='5.0'),
               MetaFeature('5<x', 'x', 'numerical', 0, lower='5.0')]
    regions = [[None, '1.0'], ['5.0', None]]
    middle = np.array([-1, 1, -1])
    assert in_direction(numbers, middle, regions, 'increase') == [
        ['5.0', None]]
    assert in_direction(numbers, middle, regions, 'decrease') == [
        [None, '1.0']]
    assert in_direction(numbers, np.array([-1, -1, 1]), regions[:1],
                        'increase') == []
    sizes = [MetaFeature('s', 'size', 'categorical', i, category=c)
             for i, c in enumerate(['s', 'm', 'l'])]
    assert in_direction(sizes, np.array([-1, 1, -1]), ['s', 'l'],
                        'increase') == ['l']
    assert in_direction(sizes, np.array([-1, 1, -1]), ['s', 'l'],
                        'decrease') == ['s']
    with pytest.raises(ValueError):  # unknown category
        in_direction(sizes, np.array([-1, -1, -1]), ['s'], 'increase')


@pytest.fixture(scope='module')
def explainer():
    rng = np.random.RandomState(1)
    X = np.column_stack([rng.uniform(0, 10, (600, 3)),
                         np.eye(3)[rng.randint(0, 3, 600)]])
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] - 2 * X[:, 5] + \
        rng.normal(0, 1, 600)
    y = np.where(score > 1, 'good', 'bad')
    clf = sklearn_tree.DecisionTreeClassifier(
        max_leaf_nodes=14, random_state=0).fit(X, y)
    spec = FeatureSpec.from_column_names(COLUMNS, features=[
        Feature('job', 'categorical', ordered=True)])
    return TreeCounterfactualExplainer(clf, spec), X


def _value(grid, feature, cell):
    values = grid.cells[feature]
    return values[cell] if feature != 'job' else cell


def test_directions(explainer):
    explainer, X = explainer
    directions = {'a': 'increase', 'b': 'decrease', 'job': 'increase'}
    for x in X[:40]:
        free = explainer.explain_instance(x)
        restricted = explainer.explain_instance(x, directions=directions)
        x_job = int(np.argmax(x[3:]))
        for found in restricted.counterfactuals:
            reference = free.at_distance(found.distance)
            assert set(found.per_leaf) <= set(reference.per_leaf)
            for leaf, changes in reference.per_leaf.items():
                kept = {}
                for feature, change in changes.items():
                    wannabe = change['wannabe']
                    if feature == 'a':
                        wannabe = [c for c in wannabe if c[0] is not None
                                   and float(c[0]) >= x[0]]
                    elif feature == 'b':
                        wannabe = [c for c in wannabe if c[1] is not None
                                   and float(c[1]) < x[1]]
                    elif feature == 'job':
                        wannabe = [c for c in wannabe
                                   if 'xyz'.index(c) > x_job]
                    kept[feature] = wannabe
                if all(kept.values()):
                    assert found.per_leaf[leaf] == {
                        f: {'is': changes[f]['is'], 'wannabe': w}
                        for f, w in kept.items()}
                else:
                    assert leaf not in found.per_leaf
        # features without a direction are not affected
        assert explainer.explain_instance(
            x, directions={'c': 'increase'}, included=['a']).ranked() == \
            explainer.explain_instance(x, included=['a']).ranked()


def test_invalid_directions(explainer):
    explainer, X = explainer
    unordered = TreeCounterfactualExplainer(explainer.tree, COLUMNS)
    with pytest.raises(ValueError):
        unordered.explain_instance(X[0], directions={'job': 'increase'})
    with pytest.raises(ValueError):
        explainer.explain_instance(X[0], directions={'a': 'up'})
    with pytest.raises(ValueError):
        explainer.explain_instance(X[0], directions={'d': 'increase'})
    with pytest.raises(TypeError):
        explainer.explain_instance(X[0], directions=['a'])


def _good(output):
    return output == 'good'


def test_monotonicity_violations(explainer):
    explainer, _ = explainer
    grid = Grid(explainer)
    leaves = explainer.tree.apply(np.array([grid.encode(p)
                                            for p in grid.points]))
    for feature in ('a', 'b', 'c', 'job'):
        i = grid.features.index(feature)
        for direction in ('increase', 'decrease'):
            found = explainer.monotonicity_violations(
                feature, direction, positive='good')
            pairs = {(v.from_leaf, v.to_leaf) for v in found}
            expected = set()
            for p, (point, output) in enumerate(zip(grid.points,
                                                    grid.outputs)):
                for q, (other, result) in enumerate(zip(grid.points,
                                                        grid.outputs)):
                    if any(a != b for k, (a, b) in enumerate(zip(point,
                                                                 other))
                           if k != i):
                        continue
                    moved = other[i] > point[i] if direction == 'increase' \
                        else other[i] < point[i]
                    if moved and output == 'good' and result != 'good':
                        expected.add((int(leaves[p]), int(leaves[q])))
            assert pairs == expected
            for v in found:
                assert (v.from_outcome, v.to_outcome) == ('good', 'bad')
                assert v.feature == feature and v.direction == direction
                assert feature not in v.context
                explainer.describe({0: v.before})
                explainer.describe({0: v.context})
                # a witness: a cell in the context and before, moved after
                witnesses = [
                    (point, other) for point in grid.points
                    for other in grid.points
                    if all(a == b for k, (a, b) in enumerate(zip(point,
                                                                 other))
                           if k != i)
                    and grid.satisfies(point, {**v.context, **v.before})
                    and grid.satisfies(other, v.after)
                    and (other[i] > point[i]) == (direction == 'increase')
                    and other[i] != point[i]]
                assert witnesses


def test_monotonicity_regression():
    # x <= 5 -> 10; 5 < x <= 10 -> 3; x > 10 -> 20
    tree = TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 0, -2, -2], threshold=[5., -2, 10., -2, -2],
        value=[[10.], [10.], [10.], [3.], [20.]], feature_names=['x'])
    explainer = TreeCounterfactualExplainer(tree)
    up = explainer.monotonicity_violations('x', 'increase')
    assert [(v.from_leaf, v.to_leaf, v.from_outcome, v.to_outcome)
            for v in up] == [(1, 3, 10., 3.)]
    assert explainer.describe({0: up[0].after})[0] == {
        'x': {'is': [], 'wannabe': ['between 5.0 and 10.0']}}
    down = explainer.monotonicity_violations('x', 'decrease')
    assert [(v.from_leaf, v.to_leaf) for v in down] == [(4, 1), (4, 3)]
    with pytest.raises(ValueError):  # no classes
        explainer.monotonicity_violations('x', positive=10.)
    with pytest.raises(ValueError):
        explainer.monotonicity_violations('x', 'sideways')
    with pytest.raises(ValueError):
        explainer.monotonicity_violations('y')


def test_monotonicity_classes(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    with pytest.raises(ValueError):  # unordered
        explainer.monotonicity_violations('job', positive='yes')
    with pytest.raises(ValueError):  # unknown class
        explainer.monotonicity_violations('age', positive='maybe')
    with pytest.raises(ValueError):  # positive is required
        explainer.monotonicity_violations('age')
    violations = explainer.monotonicity_violations('age', positive='yes')
    pairs = {(v.from_leaf, v.to_leaf) for v in violations}
    assert pairs == {(3, 6), (5, 6)}
    # a feature the tree does not use
    spec = FeatureSpec(toy_spec.columns + [('other', None)])
    tree = TreeStructure(
        toy_tree.children_left, toy_tree.children_right, toy_tree.feature,
        toy_tree.threshold, toy_tree.value, toy_tree.classes,
        n_features=toy_spec.n_columns + 1)
    assert TreeCounterfactualExplainer(tree, spec).monotonicity_violations(
        'other', positive='yes') == []
