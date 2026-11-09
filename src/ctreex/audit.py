"""
Audits of how a tree uses a feature.

* :func:`monotonicity_violations` -- pairs of leaves where changing a
  (numerical or ordered categorical) feature in a direction, with the other
  features unchanged, makes the prediction worse, e.g.,
  an applicant who would be refused a loan after their income *increases*.
* :func:`feature_flips` -- pairs of leaves where changing a feature alone
  changes the prediction, e.g., where the gender of an applicant alone
  decides whether a loan is granted
  (:meth:`ctreex.explainer.TreeCounterfactualExplainer.feature_audit` also
  counts the data points whose prediction the feature can change).

The pairs are found from the meta-features:
two leaves *differ only on a feature* when their regions are incompatible on
that feature and overlap on all the other features, so some data point in
the first leaf reaches the second one by changing that feature alone.
"""
import dataclasses

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex.distances import (allowed_rows, group_incompatibility,
                              incompatible, leaf_distance_matrix)
from ctreex.features import NUMERICAL

__all__ = ['MonotonicityViolation', 'FeatureFlip', 'FeatureAudit',
           'shared_region', 'differ_only_on', 'only_on_pairs',
           'monotonicity_violations', 'feature_flips']


@dataclasses.dataclass
class MonotonicityViolation(object):
    """
    Changing a feature in a direction makes the prediction worse.

    Data points satisfying ``context`` and ``before`` fall into
    ``from_leaf``;
    changing the feature to a value of ``after`` -- in ``direction`` --
    moves them to ``to_leaf``, whose prediction is worse.
    ``before`` (``after``) lists only the values of the feature in
    ``from_leaf`` (``to_leaf``) that some value of the other leaf lies above
    or below, as the direction requires.

    Attributes
    ----------
    feature : str
    direction : {'increase', 'decrease'}
    from_leaf, to_leaf : int
    from_outcome, to_outcome : object
        The predictions (classes or values) of the two leaves.
    before, after : dict
        ``{feature: {'is': [], 'wannabe': [...]}}`` captures the values of
        the feature in the two leaves (using the counterfactual format so
        ``TreeCounterfactualExplainer.describe({0: before})`` works).
    context : dict
        The values of the other features shared by the two leaves
        (in the same format);
        features both leaves allow fully are left out.
    """
    feature: str
    direction: str
    from_leaf: int
    to_leaf: int
    from_outcome: object
    to_outcome: object
    before: dict
    after: dict
    context: dict


def _describe(members, mask):
    """Allowed meta-features as ``wannabe`` values."""
    if members[0].kind == NUMERICAL:
        return cfs.mask_intervals(members, mask)
    return [m.category for m, keep in zip(members, mask) if keep]


def shared_region(lmf, first, second, skip=()):
    """
    The values of every feature allowed by two leaves.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    first, second : int
        Leaf ids.
    skip : collection of str
        Features to leave out.

    Returns
    -------
    dict
        ``{feature: {'is': [], 'wannabe': [...]}}`` for the features whose
        shared values are restricted (not all of them).
    """
    return _shared(lmf, _allowed_groups(lmf), lmf.leaf_index(first),
                   lmf.leaf_index(second), skip)


def _allowed_groups(lmf):
    """:func:`ctreex.distances.allowed_rows` of every group."""
    return [allowed_rows(lmf.matrix[:, indices])
            for _, _, indices in lmf.groups]


def _shared(lmf, allows, i, j, skip=(), cache=None):
    """:func:`shared_region` of the leaves in rows ``i`` and ``j``;
    ``cache`` (a dictionary) remembers the descriptions of masks."""
    region = {}
    for g, ((_, feature, indices), allowed_) in enumerate(zip(lmf.groups,
                                                              allows)):
        if feature in skip:
            continue
        mask = allowed_[i] & allowed_[j]
        if mask.all():
            continue
        key = (g, mask.tobytes())
        if cache is None or key not in cache:
            members = [lmf.meta_features[k] for k in indices]
            wannabe = _describe(members, mask)
            if cache is not None:
                cache[key] = wannabe
        else:
            wannabe = cache[key]
        region[feature] = {'is': [], 'wannabe': list(wannabe)}
    return region


def differ_only_on(lmf, feature, first, second):
    """Whether two leaves are incompatible on ``feature`` only."""
    a, b = lmf.row(first), lmf.row(second)
    for _, other, indices in lmf.groups:
        clash = incompatible(a[indices], b[indices])
        if clash != (other == feature):
            return False
    return True


def only_on_pairs(lmf, feature, matrix=None):
    """
    All the (ordered) pairs of leaves that differ only on a feature.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    feature : str
    matrix : numpy.ndarray, optional
        The leaf-to-leaf distance matrix
        (:func:`ctreex.distances.leaf_distance_matrix`, computed when not
        given).

    Returns
    -------
    numpy.ndarray
        ``(n_pairs, 2)`` row positions of the leaves (following
        ``lmf.leaf_ids``), in the order of the leaves;
        empty when the tree does not use the feature.
    """
    groups = [g for g in lmf.groups if g[1] == feature]
    if not groups:
        return np.zeros((0, 2), dtype=np.int64)
    if matrix is None:
        matrix = leaf_distance_matrix(lmf)
    # incompatible on the feature and at distance 1, i.e., on no other one
    return np.argwhere(group_incompatibility(lmf, groups[0][2])
                       & (np.asarray(matrix) == 1))


def monotonicity_violations(lmf, feature, direction, leaf_outcomes, worse,
                            matrix=None):
    """
    Pairs of leaves where changing a feature in a direction, with the other
    features unchanged, makes the prediction worse.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    feature : str
        A numerical or ordered categorical feature
        (its meta-features follow the order of its values).
    direction : {'increase', 'decrease'}
    leaf_outcomes : list
        The prediction of every leaf (following ``lmf.leaf_ids``).
    worse : callable
        ``worse(before, after)`` indicates whether the prediction ``after``
        is worse than ``before``.
    matrix : numpy.ndarray, optional
        The leaf-to-leaf distance matrix (computed when not given).

    Returns
    -------
    list of MonotonicityViolation
        In the order of the leaves;
        empty when the tree does not use the feature.
    """
    groups = [g for g in lmf.groups if g[1] == feature]
    if not groups:
        return []
    _, _, indices = groups[0]
    members = [lmf.meta_features[i] for i in indices]
    ids = lmf.leaf_ids.tolist()
    allows = allowed_rows(lmf.matrix[:, indices])
    groups_allowed, cache = _allowed_groups(lmf), {}
    violations = []
    for i, j in only_on_pairs(lmf, feature, matrix).tolist():
        first, second = ids[i], ids[j]
        before, after = np.flatnonzero(allows[i]), np.flatnonzero(allows[j])
        # the values of each leaf from (to) which the feature can move in
        # the direction to (from) a value of the other leaf
        if direction == cfs.INCREASE:
            sources, destinations = before[before < after.max()], \
                after[after > before.min()]
        else:
            sources, destinations = before[before > after.min()], \
                after[after < before.max()]
        if not sources.size or not worse(leaf_outcomes[i], leaf_outcomes[j]):
            continue
        violations.append(MonotonicityViolation(
            feature=feature, direction=direction, from_leaf=first,
            to_leaf=second, from_outcome=leaf_outcomes[i],
            to_outcome=leaf_outcomes[j],
            before={feature: {'is': [], 'wannabe': _describe(
                members, np.isin(np.arange(len(members)), sources))}},
            after={feature: {'is': [], 'wannabe': _describe(
                members, np.isin(np.arange(len(members)), destinations))}},
            context=_shared(lmf, groups_allowed, i, j, skip=(feature,),
                            cache=cache)))
    return violations


@dataclasses.dataclass
class FeatureFlip(object):
    """
    Changing a feature alone changes the prediction.

    Data points satisfying ``context`` and ``before`` fall into
    ``from_leaf``;
    changing the feature to a value of ``after`` moves them to ``to_leaf``,
    which predicts something else.

    Attributes
    ----------
    feature : str
    from_leaf, to_leaf : int
    from_outcome, to_outcome : object
        The predictions (classes or values) of the two leaves.
    before, after : dict
        ``{feature: {'is': [], 'wannabe': [...]}}`` with the values of the
        feature in the two leaves (the counterfactual format, so
        ``TreeCounterfactualExplainer.describe({0: before})`` works).
    context : dict
        The values of the other features shared by the two leaves
        (in the same format);
        features both leaves allow fully are left out.
    """
    feature: str
    from_leaf: int
    to_leaf: int
    from_outcome: object
    to_outcome: object
    before: dict
    after: dict
    context: dict


@dataclasses.dataclass
class FeatureAudit(object):
    """
    Where a feature alone changes the prediction of a tree.

    Attributes
    ----------
    feature : str
    flips : list of FeatureFlip
        All the pairs of leaves where changing the feature alone changes the
        prediction (see :func:`feature_flips`).
    affected_leaves : list of int
        The leaves whose prediction the feature alone can change
        (the ``from_leaf`` of the flips).
    unaffected_leaves : list of int
        The other leaves, whose prediction does not depend on the feature.
    n_points : int or None
        The number of audited data points (``None`` without data).
    alone : list of int or None
        The indices of the data points whose prediction changes when the
        feature alone changes (a counterfactual at distance 1).
    closest : list of int or None
        The indices of the data points whose closest counterfactuals
        (at the smallest distance with any) involve the feature.
    """
    feature: str
    flips: list
    affected_leaves: list
    unaffected_leaves: list
    n_points: int = None
    alone: list = None
    closest: list = None


def feature_flips(lmf, feature, leaf_outcomes, changed, matrix=None):
    """
    Pairs of leaves where changing a feature alone changes the prediction.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    feature : str
    leaf_outcomes : list
        The prediction of every leaf (following ``lmf.leaf_ids``).
    changed : callable
        ``changed(before, after)`` tells whether the prediction ``after`` is
        different from ``before``.
    matrix : numpy.ndarray, optional
        The leaf-to-leaf distance matrix (computed when not given).

    Returns
    -------
    list of FeatureFlip
        In the order of the leaves (both directions of every pair);
        empty when the tree does not use the feature.
    """
    groups = [g for g in lmf.groups if g[1] == feature]
    if not groups:
        return []
    _, _, indices = groups[0]
    members = [lmf.meta_features[i] for i in indices]
    ids = lmf.leaf_ids.tolist()
    allows = allowed_rows(lmf.matrix[:, indices])
    groups_allowed, cache = _allowed_groups(lmf), {}
    flips = []
    for i, j in only_on_pairs(lmf, feature, matrix).tolist():
        if not changed(leaf_outcomes[i], leaf_outcomes[j]):
            continue
        flips.append(FeatureFlip(
            feature=feature, from_leaf=ids[i], to_leaf=ids[j],
            from_outcome=leaf_outcomes[i], to_outcome=leaf_outcomes[j],
            before={feature: {'is': [], 'wannabe': _describe(members,
                                                             allows[i])}},
            after={feature: {'is': [], 'wannabe': _describe(members,
                                                            allows[j])}},
            context=_shared(lmf, groups_allowed, i, j, skip=(feature,),
                            cache=cache)))
    return flips
