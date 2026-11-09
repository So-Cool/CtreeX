"""The experimental explainer of tree ensembles."""
import json

import numpy as np
import pytest

from ctreex import (ClassProbability, Feature, FeatureSpec, MinSamples,
                    OtherClass, SameClass, TargetClass, ValueChange)
from ctreex.ensemble import (EnsembleCounterfactualExplainer,
                             EnsembleStructure, load_ensemble)

ensemble = pytest.importorskip('sklearn.ensemble')

COLUMNS = ['a', 'b', 'c', 'job_x', 'job_y', 'job_z']


def _data(n=800, seed=7):
    rng = np.random.RandomState(seed)
    X = np.column_stack([rng.uniform(0, 10, (n, 3)),
                         np.eye(3)[rng.randint(0, 3, n)]])
    score = X[:, 0] - X[:, 1] + 3 * X[:, 4] - 2 * X[:, 5] + \
        rng.normal(0, 1, n)
    return X, score


@pytest.fixture(scope='module')
def forest():
    X, score = _data()
    y = np.where(score > 3, 'high', np.where(score > -1, 'mid', 'low'))
    rf = ensemble.RandomForestClassifier(n_estimators=6, max_depth=3,
                                         random_state=0).fit(X, y)
    return rf, X


@pytest.fixture(scope='module')
def regressor():
    X, score = _data()
    reg = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=3,
                                       random_state=0).fit(X, 10 * score)
    return reg, X


def test_structure(forest, regressor):
    for model, X in (forest, regressor):
        structure = EnsembleStructure.from_sklearn(model)
        assert np.array_equal(structure.predict(X), model.predict(X)) or \
            np.allclose(structure.predict(X), model.predict(X))
        copy = load_ensemble(json.loads(structure.to_json()))
        assert np.allclose(copy._outputs(X), structure._outputs(X))
        assert copy.task == structure.task
    rf, X = forest
    assert np.allclose(EnsembleStructure.from_sklearn(rf).predict_proba(X),
                       rf.predict_proba(X))
    with pytest.raises(TypeError):
        EnsembleStructure.from_sklearn(regressor[0]).predict_proba(X)
    with pytest.raises(TypeError):  # boosting does not average its trees
        EnsembleStructure.from_sklearn(ensemble.GradientBoostingClassifier(
            n_estimators=2).fit(X, rf.predict(X)))
    with pytest.raises(ValueError):
        EnsembleStructure([])


def _cells(explainer):
    """A value inside every cell of every feature (independently of the
    explainer: the middle of the cells)."""
    cells = {}
    for kind, feature, indices in explainer.groups:
        members = [explainer.meta_features[i] for i in indices]
        values = []
        for m in members:
            if kind != 'numerical':
                values.append(m.category)
            elif m.lower is None:
                values.append(float(m.upper) - 1)
            elif m.upper is None:
                values.append(float(m.lower) + 1)
            else:
                values.append((float(m.lower) + float(m.upper)) / 2)
        cells[feature] = values
    return cells


def _set(spec, row, feature, value):
    row = row.copy()
    for j, (name, category) in enumerate(spec.columns):
        if name == feature:
            row[j] = value if category is None else float(category == value)
    return row


def _inside(value, wannabe):
    if isinstance(value, str):
        return value in wannabe
    return any((lo is None or value > float(lo)) and
               (hi is None or value <= float(hi)) for lo, hi in wannabe)


def _own(explainer, x, feature, value):
    """Whether a value lies in the cell of the data point."""
    original = explainer.spec.decode(x)[feature]
    if isinstance(value, str):
        return value == original
    for kind, name, indices in explainer.groups:
        if name == feature:
            for m in (explainer.meta_features[i] for i in indices):
                if _inside(original, [[m.lower, m.upper]]):
                    return _inside(value, [[m.lower, m.upper]])


def test_exact_counterfactuals(forest):
    rf, X = forest
    explainer = EnsembleCounterfactualExplainer(rf, COLUMNS)
    cells = _cells(explainer)
    features = list(cells)
    for x in X[:8]:
        prediction = rf.predict(x.reshape(1, -1))[0]
        explanation = explainer.explain_instance(x)
        assert explanation.prediction == prediction
        found = {c.distance: c for c in explanation.counterfactuals}
        # distance 1: every reaching cell is covered by a counterfactual of
        # its class, and nothing else
        single = set()
        for feature in features:
            for value in cells[feature]:
                if _own(explainer, x, feature, value):
                    continue
                outcome = rf.predict(_set(explainer.spec, x, feature,
                                          value).reshape(1, -1))[0]
                covered = [r for r, cf in found[1].ranked.items()
                           if feature in cf and _inside(
                               value, cf[feature]['wannabe'])] \
                    if 1 in found else []
                if outcome != prediction:
                    single.add((feature, value))
                    assert len(covered) == 1
                    assert found[1].ranked_classes[covered[0]] == [outcome]
                else:
                    assert not covered
        # distance 2: minimal pairs only
        for f, g in ((f, g) for i, f in enumerate(features)
                     for g in features[i + 1:]):
            for a in cells[f]:
                for b in cells[g]:
                    if _own(explainer, x, f, a) or _own(explainer, x, g, b):
                        continue
                    row = _set(explainer.spec, _set(explainer.spec, x, f, a),
                               g, b)
                    outcome = rf.predict(row.reshape(1, -1))[0]
                    covered = [r for r, cf in found[2].ranked.items()
                               if set(cf) == {f, g}
                               and _inside(a, cf[f]['wannabe'])
                               and _inside(b, cf[g]['wannabe'])] \
                        if 2 in found else []
                    minimal = (f, a) not in single and (g, b) not in single
                    if outcome != prediction and minimal:
                        assert len(covered) == 1
                        assert found[2].ranked_classes[covered[0]] == [
                            outcome]
                    else:
                        assert not covered


def test_targets_and_restrictions(forest):
    rf, X = forest
    explainer = EnsembleCounterfactualExplainer(rf, COLUMNS)
    x = X[0]
    prediction = rf.predict(x.reshape(1, -1))[0]
    other = [c for c in rf.classes_ if c != prediction][0]
    e = explainer.explain_instance(x, target=TargetClass(other))
    assert all(classes == [other] for _, classes, _ in
               e.ranked(with_classes=True))
    same = explainer.explain_instance(x, target=SameClass(),
                                      max_distance=1)
    assert all(classes == [prediction] for _, classes, _ in
               same.ranked(with_classes=True))
    confident = explainer.explain_instance(
        x, target=ClassProbability(other, at_least=0.9))
    for _, cf in confident.ranked():
        row = explainer.counterfactual_point(x, cf)
        assert rf.predict_proba(row.reshape(1, -1))[0][
            list(rf.classes_).index(other)] >= 0.9
    restricted = explainer.explain_instance(x, excluded=['a'])
    assert all('a' not in cf for _, cf in restricted.ranked())
    only = explainer.explain_instance(x, included=['b'])
    assert all(set(cf) == {'b'} for _, cf in only.ranked())
    for target in (MinSamples(5), OtherClass() & MinSamples(5)):
        with pytest.raises(ValueError):
            explainer.explain_instance(x, target=target)
    with pytest.raises(ValueError):
        explainer.explain_instance(x, target=TargetClass(prediction))
    with pytest.raises(ValueError):
        explainer.explain_instance(x, max_distance=3)


def test_directions(forest):
    rf, X = forest
    spec = FeatureSpec.from_column_names(COLUMNS, features=[
        Feature('job', 'categorical', ordered=True)])
    explainer = EnsembleCounterfactualExplainer(rf, spec)
    for x in X[:5]:
        e = explainer.explain_instance(x, directions={'a': 'increase',
                                                      'job': 'decrease'})
        values = spec.decode(x)
        for _, cf in e.ranked():
            if 'a' in cf:
                assert all(lo is not None and float(lo) >= values['a']
                           for lo, _ in cf['a']['wannabe'])
            if 'job' in cf:
                assert all('xyz'.index(c) < 'xyz'.index(values['job'])
                           for c in cf['job']['wannabe'])


def test_points_and_text(forest, regressor):
    rf, X = forest
    explainer = EnsembleCounterfactualExplainer(rf, COLUMNS)
    x = X[1]
    explanation = explainer.explain_instance(x)
    prediction = explanation.prediction
    explainer.spec.fit_ranges(X)
    for _, classes, cf in explanation.ranked(with_classes=True):
        rows = np.vstack([explainer.counterfactual_point(x, cf),
                          explainer.sample_counterfactual(x, cf, n=5,
                                                          random_state=0)])
        assert set(rf.predict(rows)) == set(classes)
        assert prediction not in classes
    sentences = explainer.explanation_text(explanation, n=2)
    assert sentences[0] == 'The prediction is {}.'.format(prediction)
    assert len(sentences) == 3
    reg, X = regressor
    explainer = EnsembleCounterfactualExplainer(reg, COLUMNS)
    with pytest.raises(ValueError):  # a target is required
        explainer.explain_instance(X[0])
    e = explainer.explain_instance(X[0], target=ValueChange(+20))
    base = reg.predict(X[:1])[0]
    for _, (low, high), cf in e.ranked(with_values=True):
        value = reg.predict(explainer.counterfactual_point(
            X[0], cf).reshape(1, -1))[0]
        assert value >= base + 20 and low <= value + 1e-9


def test_limits_and_unsupported(forest):
    rf, X = forest
    explainer = EnsembleCounterfactualExplainer(rf, COLUMNS,
                                                max_evaluations=10)
    with pytest.raises(ValueError, match='included'):
        explainer.explain_instance(X[0])
    for name in ('supportive_explanations', 'invariance_ranges',
                 'contextualise', 'feature_audit', 'monotonicity_violations',
                 'global_summary', 'leaf_distance_matrix',
                 'explain_contrast'):
        with pytest.raises(NotImplementedError):
            getattr(explainer, name)(X[0])
    with pytest.raises(ValueError):
        EnsembleCounterfactualExplainer(rf, COLUMNS[:-1])
    named = EnsembleCounterfactualExplainer(
        rf, COLUMNS, class_names={'low': 'L', 'mid': 'M', 'high': 'H'})
    assert named.explain_instance(X[0]).prediction in ('L', 'M', 'H')
    assert set(named.predict(X[:20])) <= {'L', 'M', 'H'}
