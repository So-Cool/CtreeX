"""
Distances between leaves and between a data point and the leaves.

Distances are computed per base feature (a *group* of meta-features) and
summed. Two (``similarity``) flavours are implemented:

* **exact** (data point vs leaf) -- for every base feature: ``1`` if the leaf's
  path contradicts the data point's value of that feature, ``0`` otherwise.
  The distance is therefore the number of features that would have to change
  for the data point to reach the leaf (a "hop" count).
* **leaf vs leaf** -- numerical features contribute the number of intervals
  that are ``1`` in one leaf and ``-1`` in the other, divided by the total
  number of meta-features; categorical features contribute the fraction of
  disagreeing completions of their partially specified one-hot vectors.
"""
import numpy as np

from ctreex.features import CATEGORICAL, NUMERICAL

__all__ = ['numerical_identity', 'exact_distance', 'numerical_distance',
           'categorical_distance', 'similarity', 'leaf_distance_matrix',
           'instance_leaf_distances']


def numerical_identity(a, b):
    """``1`` when one value is ``1`` and the other ``-1``; else ``0``."""
    return 1 if (a == -1 and b == 1) or (a == 1 and b == -1) else 0


def exact_distance(x, y):
    """
    Distance between a leaf group ``x`` and a complete data point group ``y``.

    Parameters
    ----------
    x : numpy.ndarray
        Meta-feature values of one base feature for a leaf (-1/0/1).
    y : numpy.ndarray
        Meta-feature values of the same base feature for a data point (-1/1).

    Returns
    -------
    int
        ``1`` if the leaf contradicts the data point, ``0`` otherwise.
    """
    if len(y) > 1:
        ones = np.flatnonzero(y == 1)
        if ones.size == 1:
            return 0 if x[ones[0]] in (0, 1) else 1
        if ones.size == 0:
            if np.any(y != -1):
                raise ValueError('This is not an exact (complete) vector.')
            return 1 if np.any(x == 1) else 0
        raise ValueError('This is not an exact vector -- more than one 1.')
    if len(y) == 1:
        return 0 if (x[0] == 0 or x[0] == y[0]) else 1
    raise ValueError('Exact vector of length 0.')


def numerical_distance(x, y, n_meta_features):
    """Disagreeing intervals divided by the total number of meta-features."""
    return sum(numerical_identity(i, j) for i, j in zip(x, y)) / float(
        n_meta_features)


def _differ(alpha, beta):
    return 0 if np.array_equal(alpha, beta) else 1


def _completions(v):
    zeros = np.flatnonzero(v == 0)
    if zeros.size == 0:
        return [v]
    variations = []
    for i in zeros:
        c = v.copy()
        c[i] = 1
        c[c == 0] = -1
        variations.append(c)
    return variations


def categorical_distance(x, y):
    """
    Distance between two (partial) one-hot descriptions of a feature.

    Every ``0`` (unconstrained) entry yields a completion in which that entry
    is ``1`` and the other unconstrained entries are ``-1``; the distance is
    the fraction of disagreeing pairs of completions.
    """
    if not np.any(x) or not np.any(y):
        return 0
    if _differ(x, y) == 0:
        return 0
    xs, ys = _completions(x), _completions(y)
    d = sum(_differ(i, j) for i in xs for j in ys)
    return d / float(len(xs) * len(ys))


def similarity(groups, a, b, exact=False):
    """
    Sums the per-feature distances between two meta-feature vectors.

    Parameters
    ----------
    groups : list of tuple
        ``(kind, feature, indices)`` triples (see ``LeafMetaFeatures.groups``).
    a, b : numpy.ndarray
        Meta-feature vectors; in ``exact`` mode ``a`` describes a leaf and
        ``b`` a data point.
    exact : bool
        Use :func:`exact_distance` for every feature.
    """
    distance = 0
    for kind, _, indices in groups:
        x, y = a[indices], b[indices]
        if exact:
            distance += exact_distance(x, y)
        elif kind == NUMERICAL:
            distance += numerical_distance(x, y, a.shape[0])
        elif kind == CATEGORICAL:
            distance += categorical_distance(x, y)
        else:
            raise ValueError('Unknown feature type: {!r}.'.format(kind))
    return distance


def leaf_distance_matrix(lmf):
    """
    Pairwise (symmetric) leaf-to-leaf distances.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures

    Returns
    -------
    numpy.ndarray
        ``(n_leaves, n_leaves)`` array; rows and columns follow
        ``lmf.leaf_ids``.
    """
    n = lmf.leaf_ids.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = similarity(lmf.groups, lmf.matrix[i], lmf.matrix[j])
            matrix[i, j] = matrix[j, i] = d
    return matrix


def instance_leaf_distances(lmf, instance):
    """
    Exact distances between a data point and every leaf.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-feature vector of the data point.

    Returns
    -------
    numpy.ndarray
        Integer distances following ``lmf.leaf_ids``.
    """
    return np.asarray([similarity(lmf.groups, row, instance, exact=True)
                       for row in lmf.matrix], dtype=np.int64)
