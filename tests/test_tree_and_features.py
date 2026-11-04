import json

import numpy as np
import pytest

from ctreex import Feature, FeatureSpec, TreeStructure, load_tree


@pytest.fixture
def fitted():
    sklearn = pytest.importorskip('sklearn.tree')
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.randint(18, 70, 200),
                         np.eye(3)[rng.randint(0, 3, 200)]])
    y = np.where((X[:, 0] > 40) & (X[:, 1] == 0), 'yes', 'no')
    clf = sklearn.DecisionTreeClassifier(random_state=0).fit(X, y)
    return clf, X


def test_from_sklearn_matches_apply(fitted):
    clf, X = fitted
    tree = TreeStructure.from_sklearn(clf)
    assert np.array_equal(tree.apply(X), clf.apply(X))
    assert np.array_equal(tree.predict(X), clf.predict(X))
    assert np.allclose(tree.predict_proba(X), clf.predict_proba(X))


def test_json_roundtrip(fitted, tmp_path):
    clf, X = fitted
    tree = TreeStructure.from_sklearn(clf, feature_names=['age', 'job_a',
                                                          'job_b', 'job_c'])
    path = tmp_path / 'tree.json'
    tree.to_json(str(path))
    loaded = load_tree(str(path))
    assert loaded.feature_names == ['age', 'job_a', 'job_b', 'job_c']
    assert np.array_equal(loaded.apply(X), clf.apply(X))
    assert json.loads(path.read_text())['format'] == 'ctreex-tree'


def test_pickle_loading(fitted, tmp_path):
    joblib = pytest.importorskip('joblib')
    clf, X = fitted
    joblib.dump(clf, str(tmp_path / 'model.pkl'))
    tree = load_tree(str(tmp_path / 'model.pkl'))
    assert np.array_equal(tree.apply(X), clf.apply(X))


def test_invalid_tree():
    with pytest.raises(ValueError):
        TreeStructure([1, -1], [-1, -1], [0, -2], [0.5, -2], [[1, 1], [1, 0]],
                      ['a', 'b'])
    with pytest.raises(ValueError):
        load_tree({'format': 'something else'})


def test_feature_spec():
    spec = FeatureSpec.from_column_names(
        ['age', 'job_a', 'job_b'],
        features=[Feature('job', 'categorical', display_name='job type',
                          value_names={'a': 'artist'})])
    assert spec.n_columns == 3 and spec.n_features == 2
    assert spec.features['job'].values == ['a', 'b']
    assert spec.decode([30., 0, 1]) == {'age': 30, 'job': 'b'}
    assert spec.encode({'age': 30, 'job': 'a'}).tolist() == [30., 1., 0.]
    assert spec.features['job'].describe_value('a') == 'artist'
    with pytest.raises(ValueError):
        FeatureSpec([('a', None), ('a', 'x')])
    with pytest.raises(ValueError):
        spec.decode([1., 2.])


def test_feature_spec_explicit_numerical_with_separator():
    spec = FeatureSpec.from_column_names(['loan_amount', 'job_a'],
                                         numerical=['loan_amount'])
    assert spec.columns == [('loan_amount', None), ('job', 'a')]


def test_missing_values(tmp_path):
    sklearn = pytest.importorskip('sklearn')
    from sklearn.tree import DecisionTreeClassifier
    if tuple(int(v) for v in sklearn.__version__.split('.')[:2]) < (1, 3):
        pytest.skip('Missing values need scikit-learn 1.3 or newer.')
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, (300, 2))
    y = np.where(X[:, 0] > 5, 'b', 'a')
    X[rng.rand(300) < 0.2, 0] = np.nan
    y[np.isnan(X[:, 0])] = 'b'  # missing values behave like large ones
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    tree = TreeStructure.from_sklearn(clf)
    assert tree.missing_go_to_left is not None
    X_test = np.array([[np.nan, 1.], [1., np.nan], [np.nan, np.nan],
                       [2., 3.], [8., 1.]])
    assert np.array_equal(tree.apply(X_test), clf.apply(X_test))
    assert np.array_equal(tree.predict(X), clf.predict(X))
    path = tmp_path / 'tree.json'
    tree.to_json(str(path))
    assert np.array_equal(load_tree(str(path)).apply(X_test),
                          clf.apply(X_test))
    from ctreex import TreeCounterfactualExplainer
    explainer = TreeCounterfactualExplainer(tree, ['x0', 'x1'])
    assert explainer.predict_leaf(X_test[0]) == clf.apply(X_test[:1])[0]
    with pytest.raises(ValueError):
        explainer.explain_instance(X_test[0])
    assert explainer.explain_instance(X_test[3]).prediction == 'a'


def test_missing_values_without_learnt_directions():
    # trees without missing_go_to_left send missing values to the right
    tree = TreeStructure([1, -1, -1], [2, -1, -1], [0, -2, -2], [0.5, -2, -2],
                         [[1, 1], [1, 0], [0, 1]], ['a', 'b'])
    assert tree.apply(np.array([[np.nan]])).tolist() == [2]
    assert 'missing_go_to_left' not in tree.to_dict()
