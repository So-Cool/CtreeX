"""
Concrete data points from counterfactuals.

A counterfactual describes regions -- ranges of numerical features and sets
of categories -- rather than data points.
This module turns them into data points:

* :func:`counterfactual_point` -- the point of the regions nearest to the
  data point, or the middle of the regions;
* :func:`sample_counterfactual` -- points drawn at random from the regions
  (with a seed), e.g., to share a counterfactual without revealing the
  thresholds of the tree; and
* :func:`sample_value` -- a value drawn at random from the regions of one
  feature (used by ``describe(..., hide_thresholds=True)``).

The features a counterfactual does not change keep the values of the data
point.
Numerical values are 32-bit floating point numbers (whole numbers for
integer features), so scikit-learn -- which compares 32-bit data with the
thresholds -- and :class:`ctreex.tree.TreeStructure` route them alike.
Unbounded ranges (e.g., "more than 20") are bounded by
``Feature.minimum`` and ``Feature.maximum`` (see
:meth:`ctreex.features.FeatureSpec.fit_ranges`) when sampling or taking
their middle.
"""
import math

import numpy as np

from ctreex.features import NUMERICAL

__all__ = ['above', 'at_most', 'nearest_value', 'midpoint_value',
           'sample_value', 'set_value', 'counterfactual_point',
           'sample_counterfactual']

NEAREST = 'nearest'
MIDPOINT = 'midpoint'


def above(threshold):
    """The smallest 32-bit floating point number larger than a threshold."""
    value = np.float32(threshold)
    if float(value) <= threshold:
        value = np.nextafter(value, np.float32(np.inf))
    return float(value)


def at_most(threshold):
    """The largest 32-bit floating point number not exceeding a threshold."""
    value = np.float32(threshold)
    if float(value) > threshold:
        value = np.nextafter(value, np.float32(-np.inf))
    return float(value)


def _bounds(interval):
    lower, upper = interval
    return (None if lower is None else float(lower),
            None if upper is None else float(upper))


def _distance(interval, value):
    """How far a value lies from an interval ``(lower, upper]``."""
    lower, upper = _bounds(interval)
    if lower is not None and value <= lower:
        return lower - value
    if upper is not None and value > upper:
        return value - upper
    return 0.


def _whole(interval):
    """The whole numbers of an interval ``(lower, upper]`` as
    ``[first, last]`` (``None`` for an unbounded side)."""
    lower, upper = _bounds(interval)
    first = None if lower is None else math.floor(lower) + 1
    last = None if upper is None else math.floor(upper)
    return first, last


def _clip(feature, interval):
    """An interval ``(lower, upper]`` as a closed range of values
    ``[low, high]`` within the bounds of the feature (``None`` if empty or
    unbounded)."""
    if feature.integer:
        low, high = _whole(interval)
    else:
        lower, upper = _bounds(interval)
        low = None if lower is None else above(lower)
        high = None if upper is None else at_most(upper)
    if feature.minimum is not None:
        low = feature.minimum if low is None else max(low, feature.minimum)
    if feature.maximum is not None:
        high = feature.maximum if high is None \
            else min(high, feature.maximum)
    if low is None or high is None:
        raise ValueError(
            'The range {} of {!r} is unbounded; set the minimum and maximum '
            'of the feature (e.g., FeatureSpec.fit_ranges(X)).'.format(
                interval, feature.name))
    if feature.integer:
        low, high = math.ceil(low), math.floor(high)
    return (low, high) if low <= high else None


def _closest(feature, wannabe, value):
    """The interval (category) of ``wannabe`` closest to ``value``."""
    if feature.kind != NUMERICAL:
        if feature.ordered and value in feature.values:
            position = feature.values.index(value)
            return min(wannabe, key=lambda c: abs(
                feature.values.index(c) - position))
        return wannabe[0]
    return min(wannabe, key=lambda i: _distance(i, float(value)))


def nearest_value(feature, wannabe, value):
    """
    The value of the regions of a feature nearest to a value.

    For numerical features, the bound of the closest interval, i.e., the
    smallest 32-bit number (whole number for integer features or
    ``bound + Feature.step`` for real features with a step) above its open
    lower bound or the largest one at most its upper bound.
    For categorical features, the closest category of an ordered feature,
    the first category otherwise.

    Parameters
    ----------
    feature : ctreex.features.Feature
    wannabe : list
        Intervals or categories (a counterfactual's ``'wannabe'``).
    value : object
        The value of the data point.
    """
    chosen = _closest(feature, wannabe, value)
    if feature.kind != NUMERICAL:
        return chosen
    lower, upper = _bounds(chosen)
    value = float(value)
    if upper is not None and value > upper:  # move down to the upper bound
        candidate = math.floor(upper) if feature.integer else at_most(upper)
        if lower is None or candidate > lower:
            return candidate
    if lower is not None and value <= lower:  # move up past the lower bound
        if feature.integer:
            candidate = math.floor(lower) + 1
        elif feature.step is not None:
            candidate = max(float(np.float32(lower + feature.step)),
                            above(lower))
        else:
            candidate = above(lower)
        if upper is None or candidate <= upper:
            return candidate
    raise ValueError('The range {} of {!r} has no representable value.'
                     ''.format(chosen, feature.name))


def midpoint_value(feature, wannabe, value):
    """
    The middle of the region of a feature closest to a value.

    Unbounded ranges are bounded by ``Feature.minimum`` and
    ``Feature.maximum``;
    categorical features yield the category of :func:`nearest_value`.
    """
    chosen = _closest(feature, wannabe, value)
    if feature.kind != NUMERICAL:
        return chosen
    clipped = _clip(feature, chosen)
    if clipped is None:
        raise ValueError('The range {} of {!r} lies outside of the bounds of '
                         'the feature.'.format(chosen, feature.name))
    low, high = clipped
    if feature.integer:
        return (low + high) // 2
    middle = float(np.float32((low + high) / 2))
    return min(max(middle, low), high)


def sample_value(feature, wannabe, rng):
    """
    A value drawn uniformly at random from the regions of a feature.

    For numerical features, an interval is chosen with a probability
    proportional to its length (the number of its whole numbers for integer
    features) within ``Feature.minimum`` and ``Feature.maximum``, then a value
    uniformly from it;
    for categorical features, one of the categories.

    Parameters
    ----------
    feature : ctreex.features.Feature
    wannabe : list
        Intervals or categories.
    rng : numpy.random.Generator
    """
    if feature.kind != NUMERICAL:
        return wannabe[int(rng.integers(len(wannabe)))]
    ranges = [r for r in (_clip(feature, i) for i in wannabe)
              if r is not None]
    if not ranges:
        raise ValueError('The ranges {} of {!r} lie outside of the bounds of '
                         'the feature.'.format(wannabe, feature.name))
    if feature.integer:
        sizes = np.array([high - low + 1 for low, high in ranges], float)
    else:
        sizes = np.array([high - low for low, high in ranges], float)
        if not sizes.sum():  # single values
            sizes = np.ones(len(ranges))
    low, high = ranges[int(rng.choice(len(ranges), p=sizes / sizes.sum()))]
    if feature.integer:
        return int(rng.integers(low, high + 1))
    sampled = float(np.float32(rng.uniform(low, high)))
    return min(max(sampled, low), high)


def set_value(row, spec, feature, value):
    """Writes the value of a base feature into a one-hot encoded row."""
    for j, (name, category) in enumerate(spec.columns):
        if name != feature:
            continue
        if category is None:
            row[j] = value
        else:
            row[j] = float(category == value)
    return row


def counterfactual_point(spec, x, change, how=NEAREST):
    """
    A data point satisfying a counterfactual.

    Parameters
    ----------
    spec : ctreex.features.FeatureSpec
    x : numpy.ndarray
        The one-hot encoded data point.
    change : dict
        A counterfactual (``{feature: {'is': [...], 'wannabe': [...]}}``).
    how : {'nearest', 'midpoint'}
        :func:`nearest_value` or :func:`midpoint_value` of every changed
        feature; the other features keep the values of ``x``.

    Returns
    -------
    numpy.ndarray
        One-hot encoded data point.
    """
    if how not in (NEAREST, MIDPOINT):
        raise ValueError("how has to be 'nearest' or 'midpoint'; got "
                         '{!r}.'.format(how))
    values = spec.decode(x)
    row = np.array(x, dtype=np.float64)
    choose = nearest_value if how == NEAREST else midpoint_value
    for name, part in change.items():
        set_value(row, spec, name, choose(spec.features[name],
                                          part['wannabe'], values.get(name)))
    return row


def sample_counterfactual(spec, x, change, n, rng):
    """
    Data points drawn at random from a counterfactual.

    Every changed feature is drawn with :func:`sample_value`;
    the other features keep the values of ``x``.

    Parameters
    ----------
    spec : ctreex.features.FeatureSpec
    x : numpy.ndarray
        The one-hot encoded data point.
    change : dict
        A counterfactual (``{feature: {'is': [...], 'wannabe': [...]}}``).
    n : int
        The number of points.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        ``(n, n_columns)`` one-hot encoded data points.
    """
    rows = np.tile(np.asarray(x, dtype=np.float64), (n, 1))
    for row in rows:
        for name, part in change.items():
            set_value(row, spec, name, sample_value(
                spec.features[name], part['wannabe'], rng))
    return rows
