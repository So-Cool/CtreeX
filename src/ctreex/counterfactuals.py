"""
Extraction, ranking and description of counterfactuals.

A counterfactual of a data point is a set of feature changes that move the
data point into a leaf of a different class.
Leaves are grouped by their (exact) distance to the data point -- i.e.,
the number of features that need to change -- and the conditions of every
opposite-class leaf at a given distance are turned into counterfactuals.

Numpy reimplementation of the original ``get_leaf_support_set_instance``,
``get_leaf_support_set_metric``, ``volatility_index``, ``grow_region``,
``get_contrastives``, ``transform_counterfactuals`` and ``translate``.

Data structures
---------------
``per_leaf``
    ``{leaf id: {feature: {'is': [current value], 'wannabe': [...]}}}`` --
    for numerical features ``wannabe`` holds ``[lower, upper]`` intervals
    (strings or ``None`` for unbounded), for categorical features the
    alternative categories.
``per_feature``
    ``{feature: [leaf ids]}`` -- the leaves whose counterfactual uses the
    feature.
``ranked``
    ``{rank: {feature: {'is': ..., 'wannabe': ...}}}`` -- counterfactuals in
    order of importance (see :func:`rank_counterfactuals`).
"""
import logging
import math

import numpy as np

from ctreex.distances import similarity
from ctreex.features import NUMERICAL

__all__ = ['support_set', 'support_set_metric', 'volatility_index',
           'grow_region', 'extract_counterfactuals', 'rank_counterfactuals',
           'describe_counterfactuals', 'describe_range',
           'describe_integer_range', 'format_bound', 'MissingDistance']

logger = logging.getLogger(__name__)


class MissingDistance(KeyError):
    """No leaf lies at the requested distance from the data point."""


def _unique(values, deterministic):
    if deterministic:
        return list(dict.fromkeys(values))
    return list(set(values))


def support_set(instance_class, distances, leaf_ids, leaf_classes):
    """
    Groups the leaves by their distance to a data point.

    Returns
    -------
    dict
        ``{distance: {'same': [leaf ids], 'other': [leaf ids]}}`` where
        ``same`` (``other``) lists leaves of the same (a different) class;
        distances are in the order of their first appearance.
    """
    distances = np.asarray(distances)
    out = {}
    for d in distances.tolist():
        if int(d) != d:
            raise ValueError('Exact distances have to be integers.')
        d = int(d)
        if d in out:
            continue
        selected = distances == d
        out[d] = {
            'same': leaf_ids[selected & (leaf_classes == instance_class)]
            .tolist(),
            'other': leaf_ids[selected & (leaf_classes != instance_class)]
            .tolist()}
    return out


def _metric_bins(matrix, bins_step):
    return np.arange(np.floor(np.min(matrix)),
                     np.ceil(np.max(matrix)) + bins_step, bins_step)


def support_set_metric(leaf, matrix, leaf_ids, leaf_classes, bins_step=0.5):
    """
    Groups leaves by their (binned) distance to a given leaf.

    Parameters
    ----------
    leaf : int
        The reference leaf id.
    matrix : numpy.ndarray
        Leaf-to-leaf distance matrix (see
        :func:`ctreex.distances.leaf_distance_matrix`).
    leaf_ids, leaf_classes : numpy.ndarray
        Leaf ids and classes describing the rows/columns of ``matrix``.
    bins_step : float
        Width of the distance bins.

    Returns
    -------
    dict
        ``{'(low, high]': {'same': [leaf ids], 'other': [leaf ids]}}``.
    """
    bins = _metric_bins(matrix, bins_step)
    bins[0] -= 0.0001
    labels = ['({}, {}]'.format(bins[i], bins[i + 1])
              for i in range(len(bins) - 1)]
    column = matrix[:, int(np.flatnonzero(leaf_ids == leaf)[0])]
    self_class = leaf_classes[leaf_ids == leaf][0]
    which = np.searchsorted(bins, column, side='left') - 1
    out = {}
    for b, label in enumerate(labels):
        in_bin = which == b
        out[label] = {
            'same': leaf_ids[in_bin & (leaf_classes == self_class)].tolist(),
            'other': leaf_ids[in_bin & (leaf_classes != self_class)].tolist()}
    if np.any((which < 0) | (which >= len(labels))):
        logger.warning('Undiscovered group!')
    return out


def volatility_index(leaf, matrix, leaf_ids, leaf_classes, bins_step=0.5):
    """
    Measures how close a leaf is to leaves of other classes.

    The distances from the leaf to all the leaves are binned; for bin ``k``
    (``k = 1, 2, ...``) the ratio ``(other + 1) / (same + 1)`` -- the numbers
    of other-class and same-class leaves in the bin -- is weighted by
    ``1 / k`` and summed.
    """
    bins = _metric_bins(matrix, bins_step)
    column = matrix[:, int(np.flatnonzero(leaf_ids == leaf)[0])]
    self_class = leaf_classes[leaf_ids == leaf][0]
    # pandas.cut(..., include_lowest=True) semantics
    which = np.searchsorted(bins, column, side='left') - 1
    which[column == bins[0]] = 0
    same = np.zeros(len(bins) - 1)
    other = np.zeros(len(bins) - 1)
    for w, c in zip(which, leaf_classes):
        if 0 <= w < len(bins) - 1:
            if c == self_class:
                same[w] += 1
            else:
                other[w] += 1
    ratio = (other + 1) / (same + 1)
    return float(sum(r / (k + 1) for k, r in enumerate(ratio)))


def grow_region(interval, intervals):
    """
    Merges adjacent intervals into ``interval`` (in place).

    Parameters
    ----------
    interval : list
        ``[lower, upper]`` (strings or ``None``).
    intervals : list
        Other intervals; the merged ones are removed from this list.

    Returns
    -------
    tuple
        The grown interval and the remaining intervals.
    """
    if interval[1] is not None:
        for i in intervals:
            i_l, i_u = i
            if interval[1] == i_l:
                interval[1] = i_u
                intervals.remove(i)
                return grow_region(interval, intervals)
    if interval[0] is not None:
        for i in intervals:
            i_l, i_u = i
            if interval[0] == i_u:
                interval[0] = i_l
                intervals.remove(i)
                return grow_region(interval, intervals)
    return interval, intervals


def extract_counterfactuals(lmf, instance, instance_values, support, spec,
                            distance=1, excluded=(), included=(), exact=True,
                            counterfactual=True, whole_restrictions=False,
                            target_class=None):
    """
    Extracts the counterfactuals of a data point at a given distance.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-feature vector of the data point.
    instance_values : dict
        Base feature values of the data point (reported as ``'is'``).
    support : dict
        Output of :func:`support_set`.
    spec : ctreex.features.FeatureSpec
    distance : int
        Distance of the leaves to use.
    excluded, included : collections.abc.Collection
        Base features that must not be (only these may be) part of a
        counterfactual; an empty ``included`` allows all features.
    target_class : object, optional
        Use only the leaves of this class (counterfactuals towards a specific
        class of a multi-class tree).
    whole_restrictions : bool
        How ``excluded`` and ``included`` are applied. When ``True``, a leaf
        is used only if *all* the features that have to change to reach it
        are allowed (otherwise it is skipped). When ``False`` (the original
        implementation), the forbidden features are removed from the
        counterfactual of every leaf, which leaves incomplete counterfactuals
        -- changing only the remaining features does not reach the leaf --
        and empty ones.
    exact : bool
        Distance flavour used to find the features that differ.
    counterfactual : bool
        Use leaves of another class (``True``) or of the same class.

    Returns
    -------
    per_leaf : dict
    per_feature : dict

    Raises
    ------
    MissingDistance
        No leaf lies at the given distance.
    """
    if not included:
        included = [m.feature for m in lmf.meta_features]
    side = 'other' if counterfactual else 'same'
    if distance not in support:
        raise MissingDistance(distance)
    leaves = support[distance][side]
    if target_class is not None:
        leaves = [leaf for leaf in leaves
                  if lmf.leaf_classes[lmf.leaf_index(leaf)] == target_class]

    per_leaf, per_feature = {}, {}
    for leaf in leaves:
        row = lmf.row(leaf)
        changed = [group for group in lmf.groups
                   if similarity([group], row, instance, exact) > 0]
        if whole_restrictions:
            if any(feature in excluded or feature not in included
                   for _, feature, _ in changed):
                continue  # this leaf needs a forbidden feature
        per_leaf[leaf] = {}
        for group in changed:
            kind, feature, indices = group
            if feature in excluded or feature not in included:
                continue
            values = row[indices]
            members = [lmf.meta_features[i] for i in indices]
            selected = [m for m, v in zip(members, values) if v != -1]
            if kind == NUMERICAL:
                if not selected:
                    # Legacy only (without complete_numerical_intervals): a
                    # feature split at a single threshold t has only the
                    # "A<=t" meta-feature; a leaf with "t<A" has it at -1.
                    # Swapping the bounds of the excluded intervals yields the
                    # complementary interval: (t, None) means "more than t".
                    # (Kept as tuples like the original -- ranking then
                    # treats them as categories, i.e., de-duplicates them.)
                    # With complete intervals every leaf that tests the
                    # feature has an interval at 1, so this is unreachable.
                    wannabe = [(m.upper, m.lower) for m, v in
                               zip(members, values) if v == -1]
                else:
                    intervals = [(m.lower, m.upper) for m in selected]
                    wannabe = []
                    while intervals:
                        c = list(intervals.pop())
                        c, intervals = grow_region(c, intervals)
                        wannabe.append(c)
            else:
                if not selected:  # all used categories are excluded
                    excluded_category = members[0].category
                    wannabe = [c for c in spec.features[feature].values
                               if c != excluded_category]
                else:
                    wannabe = [m.category for m in selected]
            per_leaf[leaf][feature] = {'is': [instance_values[feature]],
                                       'wannabe': wannabe}
            per_feature.setdefault(feature, []).append(leaf)
    return per_leaf, per_feature


def _merge(per_leaf, leaves, feature, deterministic_order):
    """Merges the counterfactuals of one feature across leaves."""
    current, wannabe = [], []
    numerical = False
    for leaf in leaves:
        current += per_leaf[leaf][feature]['is']
        wannabe += per_leaf[leaf][feature]['wannabe']
        if isinstance(per_leaf[leaf][feature]['wannabe'][0], list):
            numerical = True
    current = _unique(current, deterministic_order)
    if numerical:
        merged = []
        while wannabe:
            c = list(wannabe.pop())
            c, wannabe = grow_region(c, wannabe)
            merged.append(c)
        wannabe = sorted(merged, key=lambda x: float(
            x[0] if x[0] is not None else 0))
    else:
        wannabe = _unique(wannabe, deterministic_order)
    return {feature: {'is': current, 'wannabe': wannabe}}


def rank_counterfactuals(per_leaf, per_feature, distance,
                         deterministic_order=False, leaf_classes=None,
                         return_sources=False):
    """
    Orders counterfactuals by importance.

    Features are scored by the number of leaves whose counterfactual uses
    them. For ``distance == 1`` the counterfactuals of every feature are
    merged across leaves (adjacent numerical intervals are joined) and listed
    by decreasing score; for larger distances every leaf's counterfactual is
    scored with the sum of its feature scores and listed by decreasing score.

    Parameters
    ----------
    deterministic_order : bool
        Remove duplicated categories keeping their order (instead of the
        hash-dependent ``list(set(...))`` of the original implementation).
    leaf_classes : dict, optional
        ``{leaf id: class}``. When given, counterfactuals are merged and
        scored per class; at distance 1 the leaves of different classes
        are not merged, and features are scored by the leaves of the same
        class. Without it all leaves are treated alike (the original
        implementation); for two classes the results are identical.
    return_sources : bool
        Also return the leaves every ranked counterfactual comes from.

    Returns
    -------
    counterfactuals : dict
        ``{rank: {feature: {'is': [...], 'wannabe': [...]}}}``.
    sources : dict
        ``{rank: [leaf ids]}`` (only with ``return_sources``).
    """
    def class_of(leaf):
        return None if leaf_classes is None else leaf_classes[leaf]

    # (class, feature) -> leaves, in order of the features and leaves
    groups = {}
    for feature, leaves in per_feature.items():
        for leaf in leaves:
            groups.setdefault((class_of(leaf), feature), []).append(leaf)
    frequency = {key: len(leaves) for key, leaves in groups.items()}

    counterfactuals, sources = {}, {}
    if distance == 1:
        order = sorted([(frequency[key], key) for key in frequency],
                       key=lambda x: x[0], reverse=True)
        for counter, (_, key) in enumerate(order):
            counterfactuals[counter] = _merge(per_leaf, groups[key], key[1],
                                              deterministic_order)
            sources[counter] = list(groups[key])
    else:
        logger.debug('Feature frequencies: %s', frequency)
        importance = [(sum(frequency[(class_of(leaf), f)]
                           for f in per_leaf[leaf]), leaf)
                      for leaf in per_leaf]
        for counter, (_, leaf) in enumerate(
                sorted(importance, key=lambda x: x[0], reverse=True)):
            counterfactuals[counter] = per_leaf[leaf]
            sources[counter] = [leaf]
    if return_sources:
        return counterfactuals, sources
    return counterfactuals


def format_bound(bound, short=False):
    """
    Formats an interval bound (a threshold) for humans.

    Parameters
    ----------
    bound : str
        The threshold as stored in the meta-features (e.g.,
        ``'0.800000011920929'``).
    short : bool
        Use the shortest form of the largest 32-bit floating point number not
        exceeding the threshold (e.g., ``'0.8'``) when it is shorter.
        scikit-learn compares 32-bit data with the thresholds, so for its
        trees ``x <= t`` and ``x <= 0.8`` select the same data; thresholds
        that are not shorter in this form (e.g., ``'0.1'`` of a tree
        working with 64-bit data) are kept. A threshold just below a round
        number keeps that precision: ``0.8499999940395355`` becomes
        ``'0.84999996'`` (not ``'0.85'``) since the tree sends ``0.85`` to the
        right.
    """
    if not short:
        return bound
    value = float(bound)
    single = np.float32(value)
    if float(single) > value:  # round towards minus infinity
        single = np.nextafter(single, np.float32(-np.inf))
    text = str(single)
    return text if len(text) < len(bound) else bound


def describe_range(interval, short=False):
    """
    Describes an interval, e.g. ``'between 22.5 and 30.5'``.

    ``short`` formats the bounds with :func:`format_bound`.
    """
    lower, upper = interval[0], interval[1]
    if short:
        lower = None if lower is None else format_bound(lower, short=True)
        upper = None if upper is None else format_bound(upper, short=True)
    if lower is None and upper is not None:
        return 'less or equal to {}'.format(upper)
    if lower is not None and upper is None:
        return 'more than {}'.format(lower)
    if lower is not None and upper is not None:
        return 'between {} and {}'.format(lower, upper)
    raise ValueError('Incorrectly formatted interval: {!r}.'.format(interval))


def describe_integer_range(interval):
    """
    Describes an interval of an integer feature with whole numbers.

    The interval ``(lower, upper]`` of an integer feature contains the whole
    numbers from ``floor(lower) + 1`` to ``floor(upper)``; e.g.,
    ``(954.5, 1181.5]`` is described as ``'between 955 and 1181'``,
    ``(None, 883.0]`` as ``'at most 883'`` and ``(22.5, None)`` as
    ``'at least 23'``.
    """
    lower, upper = interval[0], interval[1]
    low = None if lower is None else math.floor(float(lower)) + 1
    high = None if upper is None else math.floor(float(upper))
    if low is None and high is not None:
        return 'at most {}'.format(high)
    if low is not None and high is None:
        return 'at least {}'.format(low)
    if low is not None and high is not None:
        if low == high:
            return 'exactly {}'.format(low)
        if low < high:
            return 'between {} and {}'.format(low, high)
    return describe_range(interval)


def describe_counterfactuals(counterfactuals, spec, integer_ranges=False,
                             short_thresholds=False):
    """
    Translates counterfactuals into human-readable values.

    Feature ids are replaced by ``Feature.display_name``, categories by
    ``Feature.value_names`` and intervals by :func:`describe_range`.

    Parameters
    ----------
    counterfactuals : dict
        ``{key: {feature: {'is': ..., 'wannabe': ...}}}`` (e.g. the output of
        :func:`rank_counterfactuals`).
    spec : ctreex.features.FeatureSpec
    integer_ranges : bool
        Describe the ranges of integer features (``Feature.integer``) with
        whole numbers (:func:`describe_integer_range`).
    short_thresholds : bool
        Format the thresholds of the other numerical ranges in their shortest
        form (:func:`format_bound`).
    """
    translation = {}
    for key, features in counterfactuals.items():
        translation[key] = {}
        for feature_id, cf in features.items():
            feature = spec.features[feature_id]
            if feature.kind == NUMERICAL:
                if integer_ranges and feature.integer:
                    wannabe = [describe_integer_range(i)
                               for i in cf['wannabe']]
                else:
                    wannabe = [describe_range(i, short=short_thresholds)
                               for i in cf['wannabe']]
                entry = {'is': list(cf['is']), 'wannabe': wannabe}
            else:
                entry = {'is': [feature.describe_value(i) for i in cf['is']],
                         'wannabe': [feature.describe_value(i)
                                     for i in cf['wannabe']]}
            translation[key][feature.display_name] = entry
    return translation
