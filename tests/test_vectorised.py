"""The vectorised helpers agree with their scalar counterparts."""
import itertools

import numpy as np
import pytest

from ctreex import TreeCounterfactualExplainer
from ctreex import audit, counterfactuals as cfs, distances
from ctreex.meta_features import instance_meta_features

sklearn_tree = pytest.importorskip('sklearn.tree')


@pytest.fixture(scope='module')
def explainer():
    rng = np.random.RandomState(5)
    X = np.column_stack([rng.uniform(0, 10, (800, 3)),
                         np.eye(4)[rng.randint(0, 4, 800)]])
    y = (X[:, 0] + 2 * X[:, 4] - X[:, 1] + rng.normal(0, 1, 800)) > 3
    clf = sklearn_tree.DecisionTreeClassifier(
        max_leaf_nodes=40, random_state=0).fit(X, y)
    columns = ['a', 'b', 'c', 'job_w', 'job_x', 'job_y', 'job_z']
    return TreeCounterfactualExplainer(clf, columns), X


def test_distances(explainer):
    explainer, X = explainer
    lmf = explainer.leaf_meta_features
    for _, _, indices in lmf.groups:
        block = lmf.matrix[:, indices]
        assert distances.allowed_rows(block).tolist() == \
            [distances.allowed(row).tolist() for row in block]
    matrix = distances.leaf_distance_matrix(lmf)
    for i, j in itertools.combinations(range(len(lmf.leaf_ids)), 2):
        assert matrix[i, j] == matrix[j, i] == distances.similarity(
            lmf.groups, lmf.matrix[i], lmf.matrix[j])
    for x in X[:20]:
        instance = instance_meta_features(x, explainer.meta_features)
        for exact in (True, False):
            table = distances.group_distances(lmf, instance, exact=exact)
            for i, row in enumerate(lmf.matrix):
                assert table[i].tolist() == [
                    distances.similarity([g], row, instance, exact=exact)
                    for g in lmf.groups]
        assert distances.instance_leaf_distances(lmf, instance).tolist() \
            == [distances.similarity(lmf.groups, row, instance, exact=True)
                for row in lmf.matrix]
    with pytest.raises(ValueError):
        distances.exact_distances(lmf.matrix[:, :2], np.array([1, 1]))


def _grown(intervals):
    intervals, merged = [list(i) for i in intervals], []
    while intervals:
        c = list(intervals.pop())
        c, intervals = cfs.grow_region(c, intervals)
        merged.append(c)
    return sorted(merged, key=lambda c: -np.inf if c[0] is None
                  else float(c[0]))


def test_intervals(explainer):
    explainer, _ = explainer
    lmf = explainer.leaf_meta_features
    rng = np.random.default_rng(0)
    for kind, _, indices in lmf.groups:
        if kind != 'numerical':
            continue
        members = [lmf.meta_features[i] for i in indices]
        for _ in range(50):
            mask = rng.random(len(members)) < 0.5
            selected = [(m.lower, m.upper)
                        for m, keep in zip(members, mask) if keep]
            assert cfs.mask_intervals(members, mask) == _grown(selected)
            assert cfs.merge_intervals(selected) == _grown(selected)
    assert cfs.merge_intervals([['1', '3'], ['2', '4'], [None, '0']]) == \
        [[None, '0'], ['1', '4']]
    assert cfs.merge_intervals([['1', None], ['2', '4']]) == [['1', None]]


def test_pairs(explainer):
    explainer, _ = explainer
    lmf = explainer.leaf_meta_features
    ids = lmf.leaf_ids.tolist()
    for _, feature, _ in lmf.groups:
        pairs = audit.only_on_pairs(lmf, feature).tolist()
        assert pairs == [[i, j] for i, a in enumerate(ids)
                         for j, b in enumerate(ids)
                         if i != j and audit.differ_only_on(lmf, feature,
                                                            a, b)]
    assert audit.only_on_pairs(lmf, 'unused').shape == (0, 2)


def test_leaf_index(explainer):
    explainer, _ = explainer
    lmf = explainer.leaf_meta_features
    for position, leaf in enumerate(lmf.leaf_ids):
        assert lmf.leaf_index(leaf) == position
    for unknown in (0, -1, 'x', None):
        with pytest.raises(KeyError):
            lmf.leaf_index(unknown)
