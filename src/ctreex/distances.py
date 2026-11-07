"""
Distances between leaves and between a data point and the leaves.

Distances are computed per base feature (a *group* of meta-features) and
summed;
both flavours (``similarity``) count features, so they are whole numbers:

* **exact** (data point vs leaf) -- for every base feature: ``1`` if the leaf's
  path contradicts the data point's value of that feature, ``0`` otherwise.
  The distance is therefore the number of features that would have to change
  for the data point to reach the leaf (a "hop" count).
* **leaf vs leaf** -- for every base feature: ``1`` if the two leaves allow
  no common value of the feature -- i.e., their intervals (categories) do not
  overlap -- ``0`` otherwise.
  The distance is the number of features on which the two leaves are
  incompatible.
"""
import numpy as np

__all__ = ['exact_distance', 'allowed', 'incompatible', 'similarity',
           'leaf_distance_matrix', 'instance_leaf_distances']


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


def allowed(x):
    """
    The meta-features of one base feature a leaf allows.

    A leaf allows the intervals (categories) at ``1``;
    when none is ``1``, those at ``0`` -- i.e., all of them when the leaf does
    not test the feature.

    Parameters
    ----------
    x : numpy.ndarray
        Meta-feature values of one base feature for a leaf (-1 / 0 / 1).

    Returns
    -------
    numpy.ndarray
        Boolean mask over the meta-features of the feature.
    """
    ones = x == 1
    return ones if np.any(ones) else x == 0


def incompatible(x, y):
    """
    Whether two leaves allow no common value of a base feature.

    Parameters
    ----------
    x, y : numpy.ndarray
        Meta-feature values of one base feature for two leaves (-1 / 0 / 1).

    Returns
    -------
    int
        ``1`` if the allowed intervals (categories) do not overlap, ``0``
        otherwise.
    """
    return 0 if np.any(allowed(x) & allowed(y)) else 1


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
        Use :func:`exact_distance` for every feature (otherwise
        :func:`incompatible`).
    """
    distance = 0
    for _, _, indices in groups:
        x, y = a[indices], b[indices]
        if exact:
            distance += exact_distance(x, y)
        else:
            distance += incompatible(x, y)
    return distance


def leaf_distance_matrix(lmf):
    """
    Pairwise (symmetric) leaf-to-leaf distances.

    The distance between two leaves is the number of base features on which
    they are incompatible (:func:`incompatible`);
    since the leaves partition the feature space, distinct leaves are at least
    ``1`` apart.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures

    Returns
    -------
    numpy.ndarray
        ``(n_leaves, n_leaves)`` integer array;
        rows and columns follow ``lmf.leaf_ids``.
    """
    n = lmf.leaf_ids.shape[0]
    matrix = np.zeros((n, n), dtype=np.int64)
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
