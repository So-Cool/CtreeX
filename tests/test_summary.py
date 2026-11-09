"""Global summaries, checked by brute force on the grid of tree cells."""
import numpy as np
import pytest

from ctreex import OutcomeSummary, TreeCounterfactualExplainer, TreeStructure

from test_supportive import COLUMNS, Grid  # noqa: E402

sklearn_tree = pytest.importorskip('sklearn.tree')


@pytest.fixture(scope='module', params=['classifier', 'regressor'])
def model(request):
    rng = np.random.RandomState(6)
    X = np.column_stack([rng.uniform(0, 10, (600, 3)),
                         np.eye(3)[rng.randint(0, 3, 600)]])
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] - 2 * X[:, 5] + \
        rng.normal(0, 1, 600)
    if request.param == 'classifier':
        y = np.where(score > 3, 'high', np.where(score > -1, 'mid', 'low'))
        estimator = sklearn_tree.DecisionTreeClassifier(
            max_leaf_nodes=18, random_state=0).fit(X, y)
    else:
        estimator = sklearn_tree.DecisionTreeRegressor(
            max_leaf_nodes=18, random_state=0).fit(X, 10 * score)
    return request.param, estimator, TreeCounterfactualExplainer(estimator,
                                                                 COLUMNS)


BANDS = [(None, -20), (-20, 30), (30, None)]


def _outcome(kind, output, outcome):
    if kind == 'classifier':
        return output == outcome
    low, high = outcome
    return (low is None or output >= low) and (high is None or output < high)


def test_global_summary(model):
    kind, estimator, explainer = model
    grid = Grid(explainer)
    summary = explainer.global_summary(
        bands=BANDS if kind == 'regressor' else None)
    leaves = explainer.tree.apply(np.array([grid.encode(p)
                                            for p in grid.points]))
    outcomes = list(summary)
    assert outcomes == (BANDS if kind == 'regressor'
                        else sorted(estimator.classes_.tolist()))
    for outcome, described in summary.items():
        assert isinstance(described, OutcomeSummary)
        assert described.uncovered == []
        covered = set()
        for rule in described.rules:
            assert rule.outcome == outcome
            assert set(rule.leaves) <= set(described.leaves)
            for p, (point, output) in enumerate(zip(grid.points,
                                                    grid.outputs)):
                inside = grid.satisfies(point, rule.conditions)
                if inside:
                    # exact: every cell of the rule has the outcome
                    assert _outcome(kind, output, outcome)
                    covered.add(p)
                if int(leaves[p]) in rule.leaves:
                    assert inside  # the leaves of the rule lie inside it
            explainer.describe({0: rule.conditions})
        # complete: the rules cover every cell of the outcome
        assert covered == {p for p, output in enumerate(grid.outputs)
                           if _outcome(kind, output, outcome)}
        # every leaf of the outcome is contained in some rule
        assert set(described.leaves) == {
            leaf for rule in described.rules for leaf in rule.leaves}


def test_rule_statistics(model):
    kind, estimator, explainer = model
    summary = explainer.global_summary(
        bands=BANDS if kind == 'regressor' else None)
    tree = estimator.tree_
    for outcome, described in summary.items():
        for rule in described.rules:
            assert rule.samples == int(tree.n_node_samples[rule.leaves]
                                       .sum())
            if kind == 'classifier':
                column = list(estimator.classes_).index(outcome)
                value = tree.value[rule.leaves, 0, :]
                proportions = value[:, column] / value.sum(axis=1)
                expected = (proportions * tree.n_node_samples[rule.leaves])\
                    .sum() / rule.samples
                assert rule.purity == pytest.approx(expected)
                assert rule.values is None
            else:
                values = tree.value[rule.leaves, 0, 0]
                assert rule.values == (pytest.approx(values.min()),
                                       pytest.approx(values.max()))
                assert rule.purity is None


def test_max_rules(model):
    kind, _, explainer = model
    bands = BANDS if kind == 'regressor' else None
    full = explainer.global_summary(bands=bands)
    limited = explainer.global_summary(bands=bands, max_rules=1)
    for outcome, described in limited.items():
        assert len(described.rules) == min(1, len(full[outcome].rules))
        assert described.rules[0] == full[outcome].rules[0]
        assert sorted(described.uncovered) == sorted(
            set(described.leaves) - set(described.rules[0].leaves))


def test_summary_text_and_validation(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    summary = explainer.global_summary()
    # ties go to fewer conditions; the artist rule grows up to age 50
    assert explainer.summary_text(summary) == [
        'IF age is more than 50.0 THEN the prediction is no.',
        'IF job type is baker or clerk AND age is less or equal to 30.0 THEN '
        'the prediction is no.',
        'IF age is between 30.0 and 50.0 THEN the prediction is yes.',
        'IF job type is artist AND age is less or equal to 50.0 THEN the '
        'prediction is yes.']
    assert list(explainer.global_summary(classes=['yes'])) == ['yes']
    with pytest.raises(ValueError):
        explainer.global_summary(classes=['maybe'])
    with pytest.raises(ValueError):
        explainer.global_summary(bands=[(None, 1)])
    # hand-built trees have no training statistics
    assert summary['no'].rules[0].samples is None
    assert summary['no'].rules[0].purity is None


def test_regression_validation():
    tree = TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 0, -2, -2], threshold=[10., -2, 20., -2, -2],
        value=[[150.], [100.], [200.], [150.], [300.]], feature_names=['x'])
    explainer = TreeCounterfactualExplainer(tree)
    with pytest.raises(ValueError):
        explainer.global_summary()
    with pytest.raises(ValueError):
        explainer.global_summary(classes=['a'], bands=[(None, 1)])
    with pytest.raises(ValueError):
        explainer.global_summary(bands=[(5, 1)])
    summary = explainer.global_summary(bands=[(None, 200), (200, None)])
    assert explainer.summary_text(summary) == [
        'IF x is less or equal to 20.0 THEN the prediction is less than 200.',
        'IF x is more than 20.0 THEN the prediction is at least 200.']
    everything = explainer.global_summary(bands=[(None, None)])
    assert explainer.summary_text(everything) == [
        'The prediction is always any value.']
