"""
Boolean meta-features describing the root-to-leaf paths of a decision tree.

The logical conditions of the tree splits are turned into *meta-features*:

* every category (one-hot column) of a categorical feature used by the tree
  is a meta-feature -- including the categories the tree never splits on;
* the thresholds of a numerical feature cut its range into intervals --
  e.g., ``A<=t1``, ``t1<A<=t2``, ..., ``tk<A`` (``A<=t`` and ``t<A`` for a
  single threshold) -- and every interval is a meta-feature.

Every leaf is then described by a vector over the meta-features:

* a one-hot column is ``1`` when the leaf's path requires the category,
  ``-1`` when the path excludes it and ``0`` when the path allows it among
  other categories (or does not test the feature); a path that excludes all
  but one category requires the remaining one;
* the path of a leaf confines a numerical feature to a region
  ``(lower, upper]`` (``lower``/``upper`` are the tightest bounds on the path,
  possibly infinite); an interval is ``1`` when it lies inside this region and
  ``-1`` otherwise. Since the intervals are cut at the feature's own
  thresholds, every interval is either inside or outside of the region; e.g.,
  the first interval ``A<=t1`` is ``1`` exactly when the path has no lower
  bound on the feature. All intervals are ``0`` when the path does not test the
  feature at all (unconstrained).

A data point is described in the same way with values ``1``/``-1``.

This module is a numpy reimplementation of the original (2018) code
(``tree_to_code``, ``code_to_featuresplits``, ``get_meta_features``,
``meta_features_per_leaf`` and ``instance_to_meta_features``).
"""
import dataclasses
import itertools

import numpy as np

from ctreex.features import CATEGORICAL, NUMERICAL
from ctreex.tree import REGRESSION, TREE_LEAF

__all__ = ['MetaFeature', 'Path', 'LeafMetaFeatures', 'tree_paths',
           'path_splits', 'build_meta_features', 'leaf_meta_features',
           'instance_meta_features', 'format_threshold']


def format_threshold(value):
    """Formats a threshold (``'22.5'``)."""
    return '{}'.format(float(value))


@dataclasses.dataclass(frozen=True)
class MetaFeature(object):
    """
    A boolean condition derived from the tree splits.

    Attributes
    ----------
    name : str
        ``A1_3`` (categorical) or ``A13<=22.5``, ``22.5<A13<=30.5``,
        ``30.5<A13`` (numerical).
    feature : str
        The base feature.
    kind : {'categorical', 'numerical'}
    column : int
        Index of the data column the condition refers to.
    category : str or None
        The category (categorical meta-features).
    lower, upper : str or None
        Interval bounds as strings (numerical meta-features); ``None`` means
        unbounded.
    """
    name: str
    feature: str
    kind: str
    column: int
    category: str = None
    lower: str = None
    upper: str = None

    @property
    def lower_value(self):
        return None if self.lower is None else float(self.lower)

    @property
    def upper_value(self):
        return None if self.upper is None else float(self.upper)


@dataclasses.dataclass
class Path(object):
    """
    A root-to-leaf path.

    ``conditions`` hold ``(column name, operator, value)`` triples, where the
    operator is ``'='`` for one-hot columns (value ``0`` -- left branch -- or
    ``1``) and ``'<='``/``'>'`` (value: the threshold) for numerical columns.
    ``leaf_class`` is the class of the leaf (classification trees) and
    ``leaf_value`` its prediction (regression trees);
    the other one is ``None``.
    """
    conditions: list
    leaf: int
    leaf_class: object
    leaf_value: float = None


@dataclasses.dataclass
class LeafMetaFeatures(object):
    """
    Meta-feature description of all the leaves of a tree.

    Attributes
    ----------
    meta_features : list of MetaFeature
    leaf_ids : numpy.ndarray
        Leaf ids in the order of a depth-first (left-first) traversal.
    leaf_classes : numpy.ndarray or None
        The (majority) class of every leaf (``None`` for regression trees).
    matrix : numpy.ndarray
        ``int8`` array of shape ``(n_leaves, n_meta_features)`` with values
        ``-1``, ``0`` and ``1``.
    groups : list of tuple
        ``(kind, feature, meta-feature indices)`` triples -- i.e., the
        meta-features of every base feature -- sorted by ``(kind, feature)``.
    leaf_values : numpy.ndarray or None
        The prediction of every leaf (regression trees; ``None`` for
        classification trees).
    """
    meta_features: list
    leaf_ids: np.ndarray
    leaf_classes: np.ndarray
    matrix: np.ndarray
    groups: list
    leaf_values: np.ndarray = None

    @property
    def names(self):
        return [m.name for m in self.meta_features]

    def leaf_index(self, leaf):
        """Row of the matrix describing a leaf."""
        indices = np.flatnonzero(self.leaf_ids == leaf)
        if indices.size != 1:
            raise KeyError('Unknown leaf: {}.'.format(leaf))
        return int(indices[0])

    def row(self, leaf):
        return self.matrix[self.leaf_index(leaf)]


def tree_paths(tree, spec, class_names=None):
    """
    Lists all the root-to-leaf paths of a tree (left branches first).

    Parameters
    ----------
    tree : ctreex.tree.TreeStructure
    spec : ctreex.features.FeatureSpec
    class_names : dict, optional
        Maps class labels to the names reported for the leaves.

    Returns
    -------
    list of Path

    Raises
    ------
    ValueError
        A column described as a category (one-hot column) is split at a
        threshold outside of ``[0, 1)``, i.e., it holds numbers other than
        ``0`` and ``1``.
    """
    if tree.n_features != spec.n_columns:
        raise ValueError('The tree uses {} columns but the feature '
                         'specification describes {}.'.format(
                             tree.n_features, spec.n_columns))
    paths = []
    stack = [(0, [])]
    while stack:
        node, conditions = stack.pop()
        if tree.children_left[node] == TREE_LEAF:
            if tree.task == REGRESSION:
                paths.append(Path(conditions, int(node), None,
                                  leaf_value=tree.leaf_value(node)))
                continue
            label = tree.leaf_class(node)
            if class_names is not None:
                key = label.item() if hasattr(label, 'item') else label
                label = class_names[key]
            paths.append(Path(conditions, int(node), label))
            continue
        column = int(tree.feature[node])
        name = spec.column_names[column]
        feature, category = spec.columns[column]
        if category is not None:  # one-hot column
            threshold = float(tree.threshold[node])
            if not 0 <= threshold < 1:
                raise ValueError(_misread_one_hot(spec, feature, name,
                                                  threshold))
            left, right = (name, '=', 0), (name, '=', 1)
        else:
            threshold = float(tree.threshold[node])
            left, right = (name, '<=', threshold), (name, '>', threshold)
        # right is pushed first so that the left subtree is visited first
        stack.append((int(tree.children_right[node]), conditions + [right]))
        stack.append((int(tree.children_left[node]), conditions + [left]))
    return paths


def _misread_one_hot(spec, feature, column, threshold=None):
    """Explains that numerical columns may be described as categories."""
    if threshold is None:
        problem = ('The tree requires several categories of the categorical '
                   'feature {!r} at once'.format(feature))
    else:
        problem = ('The column {!r} is described as a category of the '
                   'categorical feature {!r}, but the tree splits it at {}, '
                   'so it holds numbers other than 0 and 1'.format(
                       column, feature, threshold))
    return ('{}. Is it numerical? FeatureSpec.from_column_names treats '
            'column names containing the separator ({!r}) as '
            '<feature>{}<category>; give the numerical columns with its '
            'numerical argument or describe the columns with FeatureSpec.'
            ''.format(problem, spec.separator, spec.separator))


def path_splits(paths):
    """
    Collects the split values of every column (in order of appearance).

    Returns
    -------
    dict
        Column name -> list of distinct values used in conditions.
    """
    splits = {}
    for path in paths:
        for name, _, value in path.conditions:
            values = splits.setdefault(name, [])
            if value not in values:
                values.append(value)
    return splits


def build_meta_features(splits, spec):
    """
    Creates the meta-features from the split values of every column.

    The meta-features follow the order of the first appearance of the columns
    in the tree paths. A categorical feature gets a meta-feature for *every*
    category (in the order of the specification), not only for the categories
    the tree splits on; a numerical feature split at thresholds
    ``t1 < ... < tk`` gets the intervals ``A<=t1``, ``t1<A<=t2``, ...,
    ``tk<A`` (sorted by their thresholds).

    Parameters
    ----------
    splits : dict
        Output of :func:`path_splits`.
    spec : ctreex.features.FeatureSpec
    """
    meta_features = []
    for name, values in splits.items():
        column = spec.column_index(name)
        feature, category = spec.columns[column]
        if category is not None:
            if any(m.feature == feature for m in meta_features):
                continue  # all the categories were added already
            for value in spec.features[feature].values:
                column_name = spec.column_names[spec.columns.index(
                    (feature, value))]
                meta_features.append(MetaFeature(
                    column_name, feature, CATEGORICAL,
                    spec.column_index(column_name), category=value))
            continue
        thresholds = [format_threshold(t) for t in sorted(values)]
        bounds = [None] + thresholds + [None]
        for lower, upper in zip(bounds[:-1], bounds[1:]):
            if lower is None:
                text = '{}<={}'.format(name, upper)
            elif upper is None:
                text = '{}<{}'.format(lower, name)
            else:
                text = '{}<{}<={}'.format(lower, name, upper)
            meta_features.append(MetaFeature(
                text, feature, NUMERICAL, column, lower=lower, upper=upper))
    return meta_features


def _feature_groups(meta_features):
    """Groups meta-features by base feature, sorted by (kind, feature)."""
    groups = {}
    for i, m in enumerate(meta_features):
        groups.setdefault((m.kind, m.feature), []).append(i)
    return [(kind, feature, indices)
            for (kind, feature), indices in sorted(groups.items())]


def _numerical_bounds(conditions):
    """The tightest bounds a path puts on a numerical feature."""
    lower, upper = -np.inf, np.inf
    for _, operator, value in conditions:
        if operator == '<=':
            upper = min(upper, value)
        elif operator == '>':
            lower = max(lower, value)
        else:
            raise ValueError('Unknown numerical operator: {!r}.'.format(
                operator))
    return lower, upper


def leaf_meta_features(paths, meta_features, spec=None):
    """
    Describes every leaf with the meta-features.

    Parameters
    ----------
    paths : list of Path
    meta_features : list of MetaFeature
    spec : ctreex.features.FeatureSpec, optional
        Used to explain errors caused by numerical columns described as
        categories.

    Returns
    -------
    LeafMetaFeatures
    """
    index = {m.name: i for i, m in enumerate(meta_features)}
    by_feature = {}
    for i, m in enumerate(meta_features):
        by_feature.setdefault(m.feature, []).append(i)
    kinds = {m.feature: m.kind for m in meta_features}
    numerical_names = {m.feature for m in meta_features
                       if m.kind == NUMERICAL}

    rows = np.zeros((len(paths), len(meta_features)), dtype=np.int8)
    for r, path in enumerate(paths):
        row = [None] * len(meta_features)
        for name, group in itertools.groupby(sorted(path.conditions),
                                             lambda c: c[0]):
            group = list(group)
            if group[0][1] == '=':  # one-hot column
                for column, operator, value in group:
                    if operator != '=' or column not in index:
                        raise ValueError('Inconsistent categorical condition '
                                         'on {!r}.'.format(column))
                    if value == 0:
                        row[index[column]] = -1
                    elif value == 1:
                        row[index[column]] = 1
                    else:
                        raise ValueError('One-hot conditions have to be 0 or '
                                         '1.')
            else:
                if name not in numerical_names:
                    raise ValueError('Expected a numerical feature, got '
                                     '{!r}.'.format(name))
                lower, upper = _numerical_bounds(group)
                for i in by_feature[name]:
                    m = meta_features[i]
                    if m.lower is None:  # lower spectrum
                        row[i] = 1 if np.isinf(lower) else -1
                    elif m.upper is None:  # upper spectrum
                        row[i] = 1 if np.isinf(upper) else -1
                    else:  # middle range
                        row[i] = 1 if (lower <= m.lower_value
                                       and m.upper_value <= upper) else -1

        # complete the description of every base feature
        for feature, members in by_feature.items():
            if kinds[feature] == NUMERICAL:
                if row[members[0]] is None:  # not used on this path
                    for i in members:
                        row[i] = 0
                continue
            values = [row[i] for i in members]
            n_none = sum(v is None for v in values)
            n_ones = sum(v == 1 for v in values)
            n_minus = sum(v == -1 for v in values)
            if n_none == len(values):  # not used -> 0
                fill = 0
            elif n_ones > 1:
                if spec is None:
                    raise ValueError('One-hot error: too many 1s for '
                                     'feature {!r}.'.format(feature))
                raise ValueError(_misread_one_hot(spec, feature, None))
            elif n_ones == 1:  # one category selected -> all others are -1
                fill = -1
            elif n_none == 1 and n_ones == 0:  # all but one excluded -> 1
                fill = 1
            elif n_minus > 0 and n_ones == 0:  # several remain possible
                fill = 0
            else:  # pragma: no cover (unreachable)
                raise ValueError('Unknown categorical meta-feature pattern.')
            for i in members:
                if row[i] is None:
                    row[i] = fill
        rows[r] = row
    regression = bool(paths) and all(p.leaf_value is not None
                                     for p in paths)
    return LeafMetaFeatures(
        meta_features=list(meta_features),
        leaf_ids=np.asarray([p.leaf for p in paths], dtype=np.int64),
        leaf_classes=(None if regression
                      else np.asarray([p.leaf_class for p in paths])),
        matrix=rows, groups=_feature_groups(meta_features),
        leaf_values=(np.asarray([p.leaf_value for p in paths],
                                dtype=np.float64) if regression else None))


def instance_meta_features(x, meta_features):
    """
    Describes a (one-hot encoded) data point with the meta-features.

    Categorical meta-features are ``1`` when the corresponding column is ``1``
    and ``-1`` otherwise; numerical ones are ``1`` when the value lies in the
    interval and ``-1`` otherwise (missing -- i.e., ``nan`` -- values give
    ``-1``).

    Returns
    -------
    numpy.ndarray
        ``int8`` vector over the meta-features.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(meta_features), dtype=np.int8)
    for i, m in enumerate(meta_features):
        v = x[m.column]
        if m.kind == CATEGORICAL:
            out[i] = 1 if v == 1 else -1
        elif m.lower is None:
            out[i] = 1 if v <= m.upper_value else -1
        elif m.upper is None:
            out[i] = 1 if m.lower_value < v else -1
        else:
            out[i] = 1 if m.lower_value < v <= m.upper_value else -1
    return out
