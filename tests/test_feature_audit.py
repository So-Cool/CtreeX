"""Feature audits: where a feature alone changes the prediction."""
import numpy as np
import pytest

from ctreex import (FeatureAudit, TreeCounterfactualExplainer, TreeStructure,
                    ValueChange)

from test_supportive import COLUMNS, Grid  # noqa: E402

sklearn_tree = pytest.importorskip('sklearn.tree')


@pytest.fixture(scope='module', params=['classifier', 'regressor'])
def model(request):
    rng = np.random.RandomState(2)
    X = np.column_stack([rng.uniform(0, 10, (600, 3)),
                         np.eye(3)[rng.randint(0, 3, 600)]])
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] - 2 * X[:, 5] + \
        rng.normal(0, 1, 600)
    if request.param == 'classifier':
        estimator = sklearn_tree.DecisionTreeClassifier(
            max_leaf_nodes=14, random_state=0).fit(
                X, np.where(score > 1, 'good', 'bad'))
    else:
        estimator = sklearn_tree.DecisionTreeRegressor(
            max_leaf_nodes=14, random_state=0).fit(X, 10 * score)
    return request.param, TreeCounterfactualExplainer(estimator, COLUMNS), X


def _changed(kind, before, after):
    return before != after if kind == 'classifier' \
        else abs(after - before) > 20


def test_feature_flips(model):
    kind, explainer, X = model
    grid = Grid(explainer)
    leaves = explainer.tree.apply(np.array([grid.encode(p)
                                            for p in grid.points]))
    tolerance = None if kind == 'classifier' else 20
    for feature in ('a', 'b', 'c', 'job'):
        audit = explainer.feature_audit(feature, tolerance=tolerance)
        assert isinstance(audit, FeatureAudit) and audit.n_points is None
        i = grid.features.index(feature)
        expected = set()
        for p, point in enumerate(grid.points):
            for q, other in enumerate(grid.points):
                if all(a == b for k, (a, b) in enumerate(zip(point, other))
                       if k != i) and _changed(kind, grid.outputs[p],
                                               grid.outputs[q]):
                    expected.add((int(leaves[p]), int(leaves[q])))
        assert {(f.from_leaf, f.to_leaf) for f in audit.flips} == expected
        assert sorted(audit.affected_leaves + audit.unaffected_leaves) == \
            sorted(explainer.leaf_meta_features.leaf_ids.tolist())
        assert set(audit.affected_leaves) == {a for a, _ in expected}
        for flip in audit.flips:
            assert feature not in flip.context
            assert _changed(kind, flip.from_outcome, flip.to_outcome)
            for part in (flip.before, flip.after, flip.context):
                explainer.describe({0: part})


def test_feature_audit_with_data(model):
    kind, explainer, X = model
    grid = Grid(explainer)
    target = None if kind == 'classifier' else ValueChange(+20)
    tolerance = None if kind == 'classifier' else 20
    for feature in ('a', 'job'):
        audit = explainer.feature_audit(feature, X=X[:60], target=target,
                                        tolerance=tolerance)
        assert audit.n_points == 60
        assert set(audit.alone) <= set(audit.closest)
        i = grid.features.index(feature)
        for n, x in enumerate(X[:60]):
            prediction = explainer.tree.predict(x.reshape(1, -1))[0]
            cell = grid.cell_of(x)
            # changing the feature alone reaches the target
            reached = False
            for p, point in enumerate(grid.points):
                if all(a == b for k, (a, b) in enumerate(zip(point, cell))
                       if k != i) and point[i] != cell[i]:
                    output = grid.outputs[p]
                    reached |= output != prediction if kind == 'classifier' \
                        else output >= prediction + 20
            assert (n in audit.alone) == reached
            explanation = explainer.explain_instance(x, target=target)
            found = [c for c in explanation.counterfactuals if c.per_leaf]
            assert (n in audit.closest) == bool(
                found and feature in found[0].per_feature)


def test_feature_audit_validation(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    audit = explainer.feature_audit('job')
    # job a vs b or c decides the class of young people only
    pairs = {(f.from_leaf, f.to_leaf) for f in audit.flips}
    assert pairs == {(2, 3), (3, 2)}
    assert explainer.describe({0: audit.flips[0].context})[0] == {
        'age': {'is': [], 'wannabe': ['less or equal to 30.0']}}
    assert audit.unaffected_leaves == [5, 6]
    with pytest.raises(ValueError):
        explainer.feature_audit('jobs')
    with pytest.raises(ValueError):  # no tolerance for classifiers
        explainer.feature_audit('job', tolerance=1)
    regression = TreeStructure(
        [1, -1, -1], [2, -1, -1], [0, -2, -2], [0.5, -2, -2],
        [[1.], [0.], [2.]], feature_names=['x'])
    regression_explainer = TreeCounterfactualExplainer(regression)
    with pytest.raises(ValueError):  # regression needs a tolerance
        regression_explainer.feature_audit('x')
    with pytest.raises(ValueError):
        regression_explainer.feature_audit('x', tolerance=-1)
    assert len(regression_explainer.feature_audit('x', tolerance=1.5)
               .flips) == 2
    assert regression_explainer.feature_audit('x', tolerance=2).flips == []
    with pytest.raises(ValueError):  # data need a regression target
        regression_explainer.feature_audit('x', X=[[0.]], tolerance=1)
