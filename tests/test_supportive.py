"""Supportive explanations, invariance ranges and contextualised
counterfactuals, checked by brute force on the grid of tree cells."""
import itertools

import numpy as np
import pytest

from ctreex import (FeatureSpec, OtherClass, SameClass, TargetClass,
                    TreeCounterfactualExplainer, TreeStructure, ValueChange,
                    ValueWithin)

sklearn_tree = pytest.importorskip('sklearn.tree')

COLUMNS = ['a', 'b', 'c', 'job_x', 'job_y', 'job_z']
NUMERICAL = ['a', 'b', 'c']
JOBS = ['x', 'y', 'z']


def _data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.uniform(0, 10, (n, 3)),
                         np.eye(3)[rng.randint(0, 3, n)]])
    return X, rng


@pytest.fixture(scope='module')
def classifier():
    X, rng = _data()
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] + rng.normal(0, 1, len(X))
    y = np.where(score > 2, 'yes', np.where(X[:, 2] > 7, 'maybe', 'no'))
    clf = sklearn_tree.DecisionTreeClassifier(
        max_leaf_nodes=12, random_state=0).fit(X, y)
    return clf, X


@pytest.fixture(scope='module')
def regressor():
    X, rng = _data()
    y = 10 * X[:, 0] + 20 * X[:, 4] - 5 * X[:, 2] + rng.normal(0, 2, len(X))
    reg = sklearn_tree.DecisionTreeRegressor(
        max_leaf_nodes=12, random_state=0).fit(X, y)
    return reg, X


class Grid(object):
    """Every cell of the tree: the intervals between the thresholds of the
    numerical features and the categories of the categorical one."""

    def __init__(self, explainer):
        self.explainer = explainer
        self.tree = explainer.tree
        self.thresholds = {}
        for f, name in enumerate(NUMERICAL):
            used = self.tree.feature == f
            self.thresholds[name] = sorted(set(self.tree.threshold[used]))
        self.cells = {}
        for name in NUMERICAL:
            t = self.thresholds[name]
            reps = [t[0] - 1] if t else [0.]
            reps += [(lo + hi) / 2 for lo, hi in zip(t[:-1], t[1:])]
            reps += [t[-1] + 1] if t else []
            self.cells[name] = reps
        self.cells['job'] = JOBS
        self.features = NUMERICAL + ['job']
        self.points = list(itertools.product(
            *[range(len(self.cells[f])) for f in self.features]))
        rows = np.array([self.encode(p) for p in self.points])
        self.outputs = self.tree.predict(rows)

    def encode(self, point):
        values = {f: self.cells[f][i] for f, i in zip(self.features, point)}
        return np.array([values['a'], values['b'], values['c']] +
                        [1. if values['job'] == j else 0. for j in JOBS])

    def cell_of(self, x):
        point = []
        for f, name in enumerate(NUMERICAL):
            point.append(int(np.searchsorted(self.thresholds[name], x[f],
                                             side='left')))
        point.append(int(np.argmax(x[3:])))
        return tuple(point)

    def satisfies(self, point, conditions):
        """Whether a cell satisfies ``{feature: {'wannabe': [...]}}``."""
        for f, cell in zip(self.features, point):
            if f not in conditions:
                continue
            value = self.cells[f][cell]
            wannabe = conditions[f]['wannabe']
            if f == 'job':
                if value not in wannabe:
                    return False
            elif not any((lo is None or float(lo) < value) and
                         (hi is None or value <= float(hi))
                         for lo, hi in wannabe):
                return False
        return True

    def guaranteed(self, keep, fixed, x_cell, base=None):
        """Whether the cells agreeing with ``x_cell`` on ``fixed`` (and
        satisfying ``base``) all keep the outcome."""
        indices = [self.features.index(f) for f in fixed]
        found = False
        for point, output in zip(self.points, self.outputs):
            if any(point[i] != x_cell[i] for i in indices):
                continue
            if base is not None and not self.satisfies(point, base):
                continue
            found = True
            if not keep(output):
                return False
        return found


def _used(grid):
    return [f for f in grid.features
            if f == 'job' and any(grid.tree.feature[grid.tree.feature >= 0]
                                  >= 3)
            or f != 'job' and grid.thresholds[f]]


def _check_smallest(grid, keep, x_cell, found, candidates, base=None):
    sufficient = {}
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            sufficient[subset] = grid.guaranteed(keep, subset, x_cell, base)
    smallest = min(len(s) for s, ok in sufficient.items() if ok)
    expected = sorted(tuple(sorted(s)) for s, ok in sufficient.items()
                      if ok and len(s) == smallest)
    assert sorted(tuple(sorted(f)) for f in found) == expected


@pytest.mark.parametrize('model', ['classifier', 'regressor'])
def test_supportive_explanations(model, request):
    estimator, X = request.getfixturevalue(model)
    explainer = TreeCounterfactualExplainer(estimator, COLUMNS)
    grid = Grid(explainer)
    for x in X[:25]:
        prediction = grid.tree.predict(x.reshape(1, -1))[0]
        if model == 'classifier':
            target = None

            def keep(output):
                return output == prediction
        else:
            target = ValueWithin(15)

            def keep(output):
                return abs(output - prediction) <= 15
        x_cell = grid.cell_of(x)
        found = explainer.supportive_explanations(x, target=target)
        assert found
        for explanation in found:
            assert grid.guaranteed(keep, list(explanation.conditions),
                                   x_cell)
            # the conditions describe the cell of the data point
            assert grid.satisfies(x_cell, explanation.conditions)
            leaves = set(explanation.leaves)
            assert explainer.predict_leaf(x) in leaves
            explainer.describe({0: explanation.conditions})
        _check_smallest(grid, keep, x_cell,
                        [e.conditions for e in found], _used(grid))


@pytest.mark.parametrize('model', ['classifier', 'regressor'])
def test_invariance_ranges(model, request):
    estimator, X = request.getfixturevalue(model)
    explainer = TreeCounterfactualExplainer(estimator, COLUMNS)
    grid = Grid(explainer)
    for x in X[:25]:
        prediction = grid.tree.predict(x.reshape(1, -1))[0]
        target = None if model == 'classifier' else ValueWithin(15)
        ranges = explainer.invariance_ranges(x, target=target)
        x_cell = grid.cell_of(x)
        assert sorted(ranges) == sorted(_used(grid))
        for f, cells in grid.cells.items():
            if f not in ranges:
                continue
            i = grid.features.index(f)
            for c in range(len(cells)):
                point = x_cell[:i] + (c,) + x_cell[i + 1:]
                output = grid.outputs[grid.points.index(point)]
                kept = output == prediction if model == 'classifier' \
                    else abs(output - prediction) <= 15
                assert grid.satisfies(point, {f: ranges[f]}) == kept
        explainer.describe({0: ranges})


@pytest.mark.parametrize('model', ['classifier', 'regressor'])
def test_contextualised_counterfactuals(model, request):
    estimator, X = request.getfixturevalue(model)
    explainer = TreeCounterfactualExplainer(estimator, COLUMNS)
    grid = Grid(explainer)
    checked = 0
    for x in X[:25]:
        prediction = grid.tree.predict(x.reshape(1, -1))[0]
        if model == 'classifier':
            target = OtherClass()

            def keep(output):
                return output != prediction
        else:
            target = ValueChange(+30)

            def keep(output):
                return output >= prediction + 30
        explanation = explainer.explain_instance(x, target=target)
        contextualised = explainer.contextualise(explanation)
        x_cell = grid.cell_of(x)
        by_counterfactual = {}
        for c in contextualised:
            by_counterfactual.setdefault((c.distance, c.rank), []).append(c)
            assert c.change == explanation.at_distance(c.distance).ranked[
                c.rank]
            assert not set(c.context) & set(c.change)
            assert grid.guaranteed(keep, list(c.context), x_cell,
                                   base=c.change)
            assert grid.satisfies(x_cell, c.context)
        # every ranked counterfactual is contextualised
        assert sorted(by_counterfactual) == sorted(
            (found.distance, rank) for found in explanation.counterfactuals
            for rank in found.ranked)
        for (distance, rank), found in by_counterfactual.items():
            change = found[0].change
            candidates = [f for f in _used(grid) if f not in change]
            _check_smallest(grid, keep, x_cell, [c.context for c in found],
                            candidates, base=change)
            checked += 1
    assert checked > 10


def _off_path_tree():
    # A <= 5 -> c0 (leaf 1); A > 5: B <= 3 -> c0 (leaf 3), B > 3 -> c1
    return TreeStructure(
        [1, -1, 3, -1, -1], [2, -1, 4, -1, -1], [0, -2, 1, -2, -2],
        [5., -2, 3., -2, -2], [[1, 1], [1, 0], [1, 1], [1, 0], [0, 1]],
        ['c0', 'c1'], feature_names=['A', 'B'])


def test_explanations_beyond_the_path():
    explainer = TreeCounterfactualExplainer(_off_path_tree())
    x = np.array([2., 1.])  # leaf 1, which tests A only
    found = explainer.supportive_explanations(x)
    assert [explainer.describe({0: e.conditions})[0] for e in found] == [
        {'A': {'is': [2], 'wannabe': ['less or equal to 5.0']}},
        {'B': {'is': [1], 'wannabe': ['less or equal to 3.0']}}]
    assert [e.leaves for e in found] == [[1], [1, 3]]
    ranges = explainer.describe({0: explainer.invariance_ranges(x)})[0]
    # with A <= 5 nothing depends on B; with B <= 3 nothing depends on A
    assert ranges == {'A': {'is': [2], 'wannabe': ['any value']},
                      'B': {'is': [1], 'wannabe': ['any value']}}
    ranges = explainer.describe({0: explainer.invariance_ranges(
        np.array([7., 1.]))})[0]
    assert ranges == {'A': {'is': [7], 'wannabe': ['any value']},
                      'B': {'is': [1], 'wannabe': ['less or equal to 3.0']}}
    # a constant prediction needs no conditions
    assert [e.conditions for e in explainer.supportive_explanations(
        x, target=SameClass() | TargetClass('c1'))] == [{}]


def test_contexts():
    explainer = TreeCounterfactualExplainer(_off_path_tree())
    x = np.array([7., 1.])  # leaf 3 (c0)
    contextualised = explainer.contextualise(explainer.explain_instance(x))
    assert [(c.distance, explainer.describe({0: c.change})[0], c.context,
             c.leaves) for c in contextualised] == [
        (1, {'B': {'is': [1], 'wannabe': ['more than 3.0']}},
         {'A': {'is': [7.], 'wannabe': [['5.0', None]]}}, [4])]
    # A > 5 is required: with A <= 5 the change leads to leaf 1 (c0)
    with pytest.raises(ValueError):
        explainer.contextualise(explainer.explain_instance(x), max_size=0)


def test_targets_of_supportive_explanations():
    explainer = TreeCounterfactualExplainer(_off_path_tree())
    x = np.array([2., 1.])
    with pytest.raises(ValueError):  # does not select the own leaf
        explainer.supportive_explanations(x, target=OtherClass())
    with pytest.raises(ValueError):
        explainer.invariance_ranges(np.array([np.nan, 1.]))
    reg = TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2],
                        [0.5, -2, -2], [[1.], [0.], [2.]],
                        feature_names=['x'])
    with pytest.raises(ValueError):  # regression trees need a target
        TreeCounterfactualExplainer(reg).supportive_explanations(
            np.array([0.]))
    found = TreeCounterfactualExplainer(reg).supportive_explanations(
        np.array([0.]), target=ValueWithin(5))
    assert [e.conditions for e in found] == [{}]


def test_numerical_spec_order():
    # the conditions follow the groups of the meta-features
    clf = sklearn_tree.DecisionTreeClassifier(random_state=0).fit(
        [[0, 0], [0, 1], [1, 0], [1, 1]], ['n', 'n', 'n', 'y'])
    explainer = TreeCounterfactualExplainer(
        clf, FeatureSpec.from_column_names(['p', 'q']))
    found = explainer.supportive_explanations(np.array([1., 1.]))
    assert [sorted(e.conditions) for e in found] == [['p', 'q']]
    found = explainer.supportive_explanations(np.array([0., 1.]))
    assert [list(e.conditions) for e in found] == [['p']]
