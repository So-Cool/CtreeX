"""
Extraction, ranking and description of counterfactuals.

A counterfactual of a data point is a set of feature changes that move the
data point into a *target* leaf (see :mod:`ctreex.targets`), e.g., a leaf of
a different class.
Leaves are grouped by their (exact) distance to the data point -- i.e.,
the number of features that need to change -- and the conditions of every
target leaf at a given distance are turned into counterfactuals.

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


def _unique(values):
    """Removes duplicates keeping the order in which the values were found."""
    return list(dict.fromkeys(values))


def support_set(distances, leaf_ids, targets):
    """
    Groups the leaves by their distance to a data point.

    Parameters
    ----------
    distances : numpy.ndarray
        Exact distances between the data point and the leaves.
    leaf_ids : numpy.ndarray
        The leaf ids (following ``distances``).
    targets : numpy.ndarray
        Boolean mask of the target leaves
        (see :func:`ctreex.targets.select_targets`).

    Returns
    -------
    dict
        ``{distance: {'target': [leaf ids], 'non_target': [leaf ids]}}``
        where ``target`` (``non_target``) lists the (other) leaves selected by
        the target;
        distances are in the order of their first appearance.
    """
    distances = np.asarray(distances)
    leaf_ids = np.asarray(leaf_ids)
    targets = np.asarray(targets, dtype=bool)
    out = {}
    for d in distances.tolist():
        if int(d) != d:
            raise ValueError('Exact distances have to be integers.')
        d = int(d)
        if d in out:
            continue
        selected = distances == d
        out[d] = {'target': leaf_ids[selected & targets].tolist(),
                  'non_target': leaf_ids[selected & ~targets].tolist()}
    return out


def _leaf_column(leaf, matrix, leaf_ids, leaf_classes):
    """Distances from a leaf to all the leaves and the class of the leaf."""
    leaf_ids, leaf_classes = np.asarray(leaf_ids), np.asarray(leaf_classes)
    position = np.flatnonzero(leaf_ids == leaf)
    if position.size != 1:
        raise KeyError('Unknown leaf: {}.'.format(leaf))
    column = np.asarray(matrix)[:, int(position[0])]
    if np.any(column != np.round(column)):
        raise ValueError('Leaf-to-leaf distances have to be whole numbers.')
    return (column.astype(np.int64), leaf_ids, leaf_classes,
            leaf_classes[int(position[0])])


def support_set_metric(leaf, matrix, leaf_ids, leaf_classes):
    """
    Groups leaves by their distance to a given leaf.

    Parameters
    ----------
    leaf : int
        The reference leaf id.
    matrix : numpy.ndarray
        Leaf-to-leaf distance matrix (see
        :func:`ctreex.distances.leaf_distance_matrix`);
        the distances are whole numbers.
    leaf_ids, leaf_classes : numpy.ndarray
        Leaf ids and classes describing the rows/columns of ``matrix``.

    Returns
    -------
    dict
        ``{distance: {'same': [leaf ids], 'other': [leaf ids]}}`` for every
        distance from ``0`` (the leaf itself) to the largest distance in
        ``matrix``;
        ``same`` (``other``) lists leaves of the same (a different) class.
    """
    column, leaf_ids, leaf_classes, self_class = _leaf_column(
        leaf, matrix, leaf_ids, leaf_classes)
    out = {}
    for d in range(int(np.max(matrix)) + 1):
        at = column == d
        out[d] = {'same': leaf_ids[at & (leaf_classes == self_class)].tolist(),
                  'other': leaf_ids[at & (leaf_classes != self_class)]
                  .tolist()}
    return out


def volatility_index(leaf, matrix, leaf_ids, leaf_classes):
    """
    Measures how close a leaf is to leaves of other classes.

    For every distance ``n = 1, ..., N`` -- ``N`` being the largest distance
    in ``matrix`` -- the numbers of leaves of another class (``other``) and
    of the same class (``same``) at that distance from the leaf give the
    term ``(other + 1) / (n * (same + 1))``;
    the index is their sum.
    Large values indicate *volatile* leaves -- close to leaves of other
    classes -- and small values *stable* ones.

    Parameters
    ----------
    leaf : int
        The leaf id.
    matrix : numpy.ndarray
        Leaf-to-leaf distance matrix with whole-number distances.
    leaf_ids, leaf_classes : numpy.ndarray
        Leaf ids and classes describing the rows / columns of ``matrix``.

    Returns
    -------
    float
    """
    support = support_set_metric(leaf, matrix, leaf_ids, leaf_classes)
    return float(sum((len(leaves['other']) + 1)
                     / (n * (len(leaves['same']) + 1))
                     for n, leaves in support.items() if n > 0))


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


def extract_counterfactuals(lmf, instance, instance_values, support,
                            distance=1, excluded=(), included=(), exact=True):
    """
    Extracts the counterfactuals of a data point at a given distance.

    The counterfactuals lead to the target leaves of ``support`` at that
    distance.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    instance : numpy.ndarray
        Meta-feature vector of the data point.
    instance_values : dict
        Base feature values of the data point (reported as ``'is'``).
    support : dict
        Output of :func:`support_set`.
    distance : int
        Distance of the leaves to use.
    excluded, included : collections.abc.Collection
        Base features that must not be (only these may be) part of a
        counterfactual; an empty ``included`` allows all features.
        A leaf is used only if *all* the features that have to change to
        reach it are allowed, so every counterfactual at distance ``d``
        changes exactly ``d`` features.
    exact : bool
        Distance flavour used to find the features that differ.

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
    if distance not in support:
        raise MissingDistance(distance)
    leaves = support[distance]['target']

    per_leaf, per_feature = {}, {}
    for leaf in leaves:
        row = lmf.row(leaf)
        changed = [group for group in lmf.groups
                   if similarity([group], row, instance, exact) > 0]
        if any(feature in excluded or feature not in included
               for _, feature, _ in changed):
            continue  # this leaf needs a forbidden feature
        per_leaf[leaf] = {}
        for kind, feature, indices in changed:
            values = row[indices]
            members = [lmf.meta_features[i] for i in indices]
            # a leaf that tests a feature allows at least one of its
            # intervals (categories)
            selected = [m for m, v in zip(members, values) if v != -1]
            if kind == NUMERICAL:
                intervals = [(m.lower, m.upper) for m in selected]
                wannabe = []
                while intervals:
                    c = list(intervals.pop())
                    c, intervals = grow_region(c, intervals)
                    wannabe.append(c)
            else:
                wannabe = [m.category for m in selected]
            per_leaf[leaf][feature] = {'is': [instance_values[feature]],
                                       'wannabe': wannabe}
            per_feature.setdefault(feature, []).append(leaf)
    return per_leaf, per_feature


def _merge(per_leaf, leaves, feature):
    """Merges the counterfactuals of one feature across leaves."""
    current, wannabe = [], []
    numerical = False
    for leaf in leaves:
        current += per_leaf[leaf][feature]['is']
        wannabe += per_leaf[leaf][feature]['wannabe']
        if isinstance(per_leaf[leaf][feature]['wannabe'][0], list):
            numerical = True
    current = _unique(current)
    if numerical:
        merged = []
        while wannabe:
            c = list(wannabe.pop())
            c, wannabe = grow_region(c, wannabe)
            merged.append(c)
        # unbounded-below intervals first, the others by their lower bound
        wannabe = sorted(merged, key=lambda x: (
            -np.inf if x[0] is None else float(x[0])))
    else:
        wannabe = _unique(wannabe)
    return {feature: {'is': current, 'wannabe': wannabe}}


def rank_counterfactuals(per_leaf, per_feature, distance, leaf_classes=None,
                         return_sources=False):
    """
    Orders counterfactuals by importance.

    Features are scored by the number of leaves whose counterfactual uses
    them. For ``distance == 1`` the counterfactuals of every feature are
    merged across leaves (adjacent numerical intervals are joined) and listed
    by decreasing score; for larger distances every leaf's counterfactual is
    scored with the sum of its feature scores and listed by decreasing score.

    Duplicated values are removed keeping the order in which they were found.

    Parameters
    ----------
    leaf_classes : dict, optional
        ``{leaf id: class}``. Counterfactuals are merged and scored per
        class: at distance 1 the leaves of different classes are not merged,
        and features are scored by the leaves of the same class. Without it
        all leaves are treated as one class.
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
            counterfactuals[counter] = _merge(per_leaf, groups[key], key[1])
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


def describe_counterfactuals(counterfactuals, spec, integer_ranges=False):
    """
    Translates counterfactuals into human-readable values.

    Feature ids are replaced by ``Feature.display_name``, categories by
    ``Feature.value_names`` and intervals by :func:`describe_range` with
    the thresholds in their shortest form (:func:`format_bound`).

    Parameters
    ----------
    counterfactuals : dict
        ``{key: {feature: {'is': ..., 'wannabe': ...}}}`` (e.g. the output of
        :func:`rank_counterfactuals`).
    spec : ctreex.features.FeatureSpec
    integer_ranges : bool
        Describe the ranges of integer features (``Feature.integer``) with
        whole numbers (:func:`describe_integer_range`).
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
                    wannabe = [describe_range(i, short=True)
                               for i in cf['wannabe']]
                entry = {'is': list(cf['is']), 'wannabe': wannabe}
            else:
                entry = {'is': [feature.describe_value(i) for i in cf['is']],
                         'wannabe': [feature.describe_value(i)
                                     for i in cf['wannabe']]}
            translation[key][feature.display_name] = entry
    return translation
