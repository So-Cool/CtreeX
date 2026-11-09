"""
Audits of how a tree uses a feature.

* :func:`monotonicity_violations` -- pairs of leaves where changing a
  (numerical or ordered categorical) feature in a direction, with the other
  features unchanged, makes the prediction worse, e.g.,
  an applicant who would be refused a loan after their income *increases*.

The pairs are found from the meta-features:
two leaves *differ only on a feature* when their regions are incompatible on
that feature and overlap on all the other features, so some data point in
the first leaf reaches the second one by changing that feature alone.
"""
import dataclasses

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex.distances import allowed, incompatible
from ctreex.features import NUMERICAL

__all__ = ['MonotonicityViolation', 'shared_region', 'differ_only_on',
           'monotonicity_violations']


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
    kept = [m for m, keep in zip(members, mask) if keep]
    if not kept:
        return []
    if kept[0].kind == NUMERICAL:
        intervals = [(m.lower, m.upper) for m in kept]
        wannabe = []
        while intervals:
            c = list(intervals.pop())
            c, intervals = cfs.grow_region(c, intervals)
            wannabe.append(c)
        return sorted(wannabe, key=lambda c: -np.inf if c[0] is None
                      else float(c[0]))
    return [m.category for m in kept]


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
    a, b = lmf.row(first), lmf.row(second)
    region = {}
    for _, feature, indices in lmf.groups:
        if feature in skip:
            continue
        mask = allowed(a[indices]) & allowed(b[indices])
        if not mask.all():
            members = [lmf.meta_features[i] for i in indices]
            region[feature] = {'is': [], 'wannabe': _describe(members, mask)}
    return region


def differ_only_on(lmf, feature, first, second):
    """Whether two leaves are incompatible on ``feature`` only."""
    a, b = lmf.row(first), lmf.row(second)
    for _, other, indices in lmf.groups:
        clash = incompatible(a[indices], b[indices])
        if clash != (other == feature):
            return False
    return True


def monotonicity_violations(lmf, feature, direction, leaf_outcomes, worse):
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
    positions = {}
    for leaf in ids:
        mask = allowed(lmf.row(leaf)[indices])
        positions[leaf] = (mask, np.flatnonzero(mask))
    violations = []
    for i, first in enumerate(ids):
        for j, second in enumerate(ids):
            if first == second:
                continue
            before, after = positions[first][1], positions[second][1]
            # the values of each leaf from (to) which the feature can move in
            # the direction to (from) a value of the other leaf
            if direction == cfs.INCREASE:
                sources, destinations = before[before < after.max()], \
                    after[after > before.min()]
            else:
                sources, destinations = before[before > after.min()], \
                    after[after < before.max()]
            if not sources.size or not worse(leaf_outcomes[i],
                                             leaf_outcomes[j]):
                continue
            if not differ_only_on(lmf, feature, first, second):
                continue
            violations.append(MonotonicityViolation(
                feature=feature, direction=direction, from_leaf=first,
                to_leaf=second, from_outcome=leaf_outcomes[i],
                to_outcome=leaf_outcomes[j],
                before={feature: {'is': [], 'wannabe': _describe(
                    members, np.isin(np.arange(len(members)), sources))}},
                after={feature: {'is': [], 'wannabe': _describe(
                    members, np.isin(np.arange(len(members)),
                                     destinations))}},
                context=shared_region(lmf, first, second, skip=(feature,))))
    return violations
