"""
Supportive explanations, invariance ranges and contextualised counterfactuals.

Counterfactuals say what has to change for the prediction to change;
the explanations in this module say what guarantees an outcome:

* a *supportive explanation* of a data point is a smallest set of its
  features that guarantees the prediction (more precisely, a target, e.g.,
  ``SameClass()``) whatever the values of the other features --
  "since the duration is at most 11.5 months and the account is A14, the
  loan is accepted";
* the *invariance range* of a feature lists its values that keep the
  prediction when the other features do not change; and
* a *contextualised counterfactual* adds to a counterfactual a smallest set
  of the other features that guarantees its outcome whatever the values of
  the remaining features --
  "had the duration been at most 11.5 months, the loan would have been
  accepted, provided that the account is A14".

All of them are computed exactly from the meta-features.
A feature *fixed* at the value of the data point is described by the
meta-feature of that value: the interval between the two thresholds around
it (numerical features) or its category.
A leaf is *consistent* with a set of conditions when its region overlaps
them, i.e., some data point satisfying the conditions may fall into it;
conditions guarantee an outcome when all the leaves consistent with them are
target leaves.
"""
import dataclasses
import itertools

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex.distances import allowed, exact_distance
from ctreex.features import NUMERICAL

__all__ = ['SupportiveExplanation', 'ContextualisedCounterfactual',
           'fixed_consistency', 'region_consistency', 'tested_features',
           'supportive_explanations', 'invariance_ranges',
           'contextualise_counterfactual', 'matching_features']


@dataclasses.dataclass
class SupportiveExplanation(object):
    """
    A smallest set of features of a data point that guarantees an outcome.

    Attributes
    ----------
    conditions : dict
        ``{feature: {'is': [value], 'wannabe': [interval or category]}}``
        -- every feature fixed at the meta-feature of the data point's value
        (the format of the counterfactuals, so
        ``TreeCounterfactualExplainer.describe({0: conditions})`` works);
        empty when the outcome does not depend on any feature.
    leaves : list of int
        The leaves consistent with the conditions (all of them are target
        leaves).
    """
    conditions: dict
    leaves: list


@dataclasses.dataclass
class ContextualisedCounterfactual(object):
    """
    A counterfactual with a smallest set of the other features that
    guarantees its outcome.

    Attributes
    ----------
    distance : int
        The distance of the counterfactual.
    rank : int
        Its rank at that distance.
    change : dict
        The counterfactual
        (``{feature: {'is': [...], 'wannabe': [...]}}``).
    context : dict
        The other features fixed at the meta-features of the data point's
        values (the format of the counterfactuals);
        empty when the change alone guarantees the outcome.
    leaves : list of int
        The leaves consistent with the change and the context (all of them
        are target leaves).
    """
    distance: int
    rank: int
    change: dict
    context: dict
    leaves: list


def fixed_consistency(lmf, instance):
    """
    Which leaves are consistent with every feature fixed at a data point.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-features of the data point.

    Returns
    -------
    dict
        ``{feature: boolean mask over the leaves}`` -- the leaves whose
        region contains the meta-feature of the data point's value.
    """
    return {feature: np.array([exact_distance(row[indices],
                                              instance[indices]) == 0
                               for row in lmf.matrix])
            for _, feature, indices in lmf.groups}


def _bound(value, unbounded):
    return unbounded if value is None else float(value)


def _region_mask(kind, members, wannabe):
    """The meta-features of a feature inside a ``wannabe`` list."""
    if kind == NUMERICAL:
        intervals = [(_bound(lower, -np.inf), _bound(upper, np.inf))
                     for lower, upper in wannabe]
        return np.array([any(lower <= _bound(m.lower, -np.inf)
                             and _bound(m.upper, np.inf) <= upper
                             for lower, upper in intervals)
                         for m in members])
    return np.array([m.category in wannabe for m in members])


def region_consistency(lmf, change):
    """
    Which leaves are consistent with a counterfactual change.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    change : dict
        ``{feature: {'is': [...], 'wannabe': [...]}}``.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the leaves: the leaves that allow some value of
        every changed feature in its ``wannabe`` range.
    """
    mask = np.ones(lmf.leaf_ids.shape[0], dtype=bool)
    for kind, feature, indices in lmf.groups:
        if feature not in change:
            continue
        members = [lmf.meta_features[i] for i in indices]
        region = _region_mask(kind, members, change[feature]['wannabe'])
        mask &= np.array([np.any(allowed(row[indices]) & region)
                          for row in lmf.matrix])
    return mask


def tested_features(lmf, leaves):
    """
    The features tested by any of the leaves (in the order of the groups).

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    leaves : numpy.ndarray
        Boolean mask over the leaves.
    """
    rows = lmf.matrix[leaves]
    return [feature for _, feature, indices in lmf.groups
            if np.any(rows[:, indices] != 0)]


def _condition(lmf, instance, instance_values, feature):
    """A feature fixed at the meta-feature of the data point's value."""
    kind, _, indices = next(g for g in lmf.groups if g[1] == feature)
    if kind == NUMERICAL:
        i = indices[int(np.flatnonzero(instance[indices] == 1)[0])]
        m = lmf.meta_features[i]
        wannabe = [[m.lower, m.upper]]
    else:
        wannabe = [instance_values[feature]]
    return {'is': [instance_values[feature]], 'wannabe': wannabe}


def _smallest(candidates, base, fixed, targets, max_size):
    """All the smallest subsets of ``candidates`` that, fixed at the data
    point on top of ``base``, guarantee the target, with their leaves."""
    for size in range(min(max_size, len(candidates)) + 1):
        found = []
        for subset in itertools.combinations(candidates, size):
            mask = base.copy()
            for feature in subset:
                mask &= fixed[feature]
            if mask.any() and targets[mask].all():
                found.append((subset, mask))
        if found:
            return found
    return []


def supportive_explanations(lmf, leaf, instance, instance_values, targets):
    """
    All the smallest supportive explanations of a data point.

    The candidates are all the features used by the tree (an explanation may
    include features that the path of the data point does not test).
    Fixing the features tested on that path leaves only the leaf of the data
    point, so the smallest explanations have at most as many features;
    the search tries the sets of candidates by increasing size.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    leaf : int
        The leaf of the data point; it has to be a target leaf.
    instance : numpy.ndarray
        Meta-features of the data point.
    instance_values : dict
        Base feature values of the data point (reported as ``'is'``).
    targets : numpy.ndarray
        Boolean mask of the target leaves.

    Returns
    -------
    list of SupportiveExplanation
    """
    targets = np.asarray(targets, dtype=bool)
    own = lmf.leaf_ids == leaf
    if not targets[own].all():
        raise ValueError('Supportive explanations guarantee a target that '
                         'selects the leaf of the data point, e.g., '
                         'SameClass() (or ValueWithin(...) for regression '
                         'trees).')
    fixed = fixed_consistency(lmf, instance)
    every = np.ones_like(targets)
    found = _smallest(tested_features(lmf, every), every, fixed, targets,
                      len(tested_features(lmf, own)))
    return [SupportiveExplanation(
        conditions={f: _condition(lmf, instance, instance_values, f)
                    for f in subset},
        leaves=lmf.leaf_ids[mask].tolist()) for subset, mask in found]


def invariance_ranges(lmf, instance, instance_values, targets):
    """
    The values of every feature that keep the outcome when the other features
    do not change.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-features of the data point.
    instance_values : dict
        Base feature values of the data point (reported as ``'is'``).
    targets : numpy.ndarray
        Boolean mask of the target leaves.

    Returns
    -------
    dict
        ``{feature: {'is': [value], 'wannabe': [intervals or categories]}}``
        for every feature used by the tree (the format of the
        counterfactuals): the ranges (categories) of values -- including the
        one of the data point when its leaf is a target leaf -- for which the
        data point reaches a target leaf.
        Features not used by the tree do not affect the outcome.
    """
    targets = np.asarray(targets, dtype=bool)
    fixed = fixed_consistency(lmf, instance)
    ranges = {}
    for kind, feature, indices in lmf.groups:
        others = np.ones_like(targets)
        for other, mask in fixed.items():
            if other != feature:
                others &= mask
        leaf_allows = np.array([allowed(row[indices]) for row in lmf.matrix])
        kept = []
        for position, i in enumerate(indices):
            mask = others & leaf_allows[:, position]
            if mask.any() and targets[mask].all():
                kept.append(lmf.meta_features[i])
        if kind == NUMERICAL:
            intervals = [(m.lower, m.upper) for m in kept]
            wannabe = []
            while intervals:
                c = list(intervals.pop())
                c, intervals = cfs.grow_region(c, intervals)
                wannabe.append(c)
            wannabe.sort(key=lambda c: -np.inf if c[0] is None
                         else float(c[0]))
        else:
            wannabe = [m.category for m in kept]
        ranges[feature] = {'is': [instance_values[feature]],
                           'wannabe': wannabe}
    return ranges


def contextualise_counterfactual(lmf, instance, instance_values, targets,
                                 change, max_size):
    """
    All the smallest contexts of a counterfactual.

    The changed features are restricted to their counterfactual ranges;
    the candidate context features are the other features tested by the
    leaves consistent with the change.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-features of the data point.
    instance_values : dict
        Base feature values of the data point (reported as ``'is'``).
    targets : numpy.ndarray
        Boolean mask of the target leaves.
    change : dict
        The counterfactual (``{feature: {'is': [...], 'wannabe': [...]}}``).
    max_size : int
        The largest number of context features.

    Returns
    -------
    list of tuple
        ``(context, leaves)`` pairs; empty when there is no context of at
        most ``max_size`` features.
    """
    targets = np.asarray(targets, dtype=bool)
    base = region_consistency(lmf, change)
    fixed = fixed_consistency(lmf, instance)
    candidates = [f for f in tested_features(lmf, base) if f not in change]
    return [({f: _condition(lmf, instance, instance_values, f)
              for f in subset}, lmf.leaf_ids[mask].tolist())
            for subset, mask in _smallest(candidates, base, fixed, targets,
                                          max_size)]


def matching_features(lmf, instance, change):
    """
    The changed features whose new values include the value of a data point.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-features of the data point (e.g., a reference data point).
    change : dict
        A counterfactual (``{feature: {'is': [...], 'wannabe': [...]}}``).

    Returns
    -------
    list of str
        In the order of ``change``.
    """
    matching = []
    for kind, feature, indices in lmf.groups:
        if feature not in change:
            continue
        members = [lmf.meta_features[i] for i in indices]
        region = _region_mask(kind, members, change[feature]['wannabe'])
        if np.any(region & (instance[indices] == 1)):
            matching.append(feature)
    return [f for f in change if f in matching]
