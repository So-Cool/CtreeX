"""
Counterfactual explanations of decision tree predictions.

:class:`TreeCounterfactualExplainer` is initialised with a fitted model and
a description of its features, validates its input, and explains individual
data points with :meth:`TreeCounterfactualExplainer.explain_instance`.

The algorithm:

1. the conditions on the root-to-leaf paths are turned into boolean
   meta-features (:mod:`ctreex.meta_features`);
2. the data point is described with the same meta-features and its exact
   distance to every leaf -- i.e., the number of features that would need to
   change to reach the leaf -- is computed (:mod:`ctreex.distances`);
3. the conditions of leaves of the opposite class at distance 1, 2, ... are
   turned into counterfactuals and ranked (:mod:`ctreex.counterfactuals`).
"""
import dataclasses
import logging
import warnings

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex import distances
from ctreex import meta_features as mf
from ctreex import _validation as val
from ctreex.features import FeatureSpec
from ctreex.tree import load_tree

__all__ = ['TreeCounterfactualExplainer', 'CounterfactualExplanation',
           'DistanceCounterfactuals']

logger = logging.getLogger(__name__)


def _python(value):
    """Converts numpy scalars (e.g., class labels) to Python objects."""
    return value.item() if isinstance(value, np.generic) else value


@dataclasses.dataclass
class DistanceCounterfactuals(object):
    """
    Counterfactuals of a data point at a given distance.

    Attributes
    ----------
    distance : int
        The number of features changed by every counterfactual.
    per_leaf : dict
        ``{leaf id: {feature: {'is': [...], 'wannabe': [...]}}}``.
    per_feature : dict
        ``{feature: [leaf ids]}``.
    ranked : dict
        ``{rank: {feature: {'is': [...], 'wannabe': [...]}}}``.
    leaf_classes : dict
        ``{leaf id: class}`` -- the class every leaf of ``per_leaf`` predicts.
    ranked_classes : dict
        ``{rank: [classes]}`` -- the classes a ranked counterfactual leads to
        (counterfactuals are merged per class, so this is a single class).
    """
    distance: int
    per_leaf: dict
    per_feature: dict
    ranked: dict
    leaf_classes: dict = dataclasses.field(default_factory=dict)
    ranked_classes: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CounterfactualExplanation(object):
    """
    Counterfactual explanation of a single prediction.

    Attributes
    ----------
    leaf : int
        The leaf the data point falls into.
    prediction : object
        The predicted class.
    instance_values : dict
        Base feature values of the data point.
    meta_features : numpy.ndarray
        The data point described with the meta-features.
    leaf_distances : numpy.ndarray
        Exact distances to all the leaves (in ``LeafMetaFeatures`` order).
    support : dict
        ``{distance: {'same': [...], 'other': [...]}}`` leaf groups.
    counterfactuals : list of DistanceCounterfactuals
        Counterfactuals ordered by increasing distance.
    counterfactual : bool
        Whether the counterfactuals are class-contrastive (``True``) or
        class-equivalent (``False``).
    """
    leaf: int
    prediction: object
    instance_values: dict
    meta_features: np.ndarray
    leaf_distances: np.ndarray
    support: dict
    counterfactuals: list
    counterfactual: bool = True

    @property
    def distances(self):
        return [c.distance for c in self.counterfactuals]

    def at_distance(self, distance):
        """Counterfactuals at the given distance (``None`` if absent)."""
        for c in self.counterfactuals:
            if c.distance == distance:
                return c
        return None

    def ranked(self, with_classes=False):
        """
        All ranked counterfactuals.

        Returns
        -------
        list
            ``(distance, counterfactual)`` pairs, or ``(distance, classes,
            counterfactual)`` triples with ``with_classes``.
        """
        if with_classes:
            return [(c.distance, c.ranked_classes[rank], cf)
                    for c in self.counterfactuals
                    for rank, cf in c.ranked.items()]
        return [(c.distance, cf) for c in self.counterfactuals
                for cf in c.ranked.values()]


class TreeCounterfactualExplainer(object):
    """
    Explains decision tree predictions with counterfactuals.

    Parameters
    ----------
    model : TreeStructure, scikit-learn tree, str or dict
        The tree, i.e., anything accepted by :func:`ctreex.tree.load_tree`.
    feature_spec : FeatureSpec or list of str, optional
        Description of the data columns. A list of column names is converted
        with :meth:`FeatureSpec.from_column_names`; when omitted the feature
        names stored in the tree are used.
    class_names : dict or list, optional
        Names of the classes (a dictionary mapping class labels to names or a
        list following ``model.classes``); used for predictions and leaves.

    Attributes
    ----------
    tree : TreeStructure
    spec : FeatureSpec
    paths : list of ctreex.meta_features.Path
    meta_features : list of ctreex.meta_features.MetaFeature
    leaf_meta_features : ctreex.meta_features.LeafMetaFeatures

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    >>> explainer = TreeCounterfactualExplainer(clf, ['age', 'job_a', 'job_b'])
    >>> explanation = explainer.explain_instance(X[0])
    """

    def __init__(self, model, feature_spec=None, class_names=None):
        self.tree = load_tree(model)
        if feature_spec is None:
            if self.tree.feature_names is None:
                raise ValueError('The model does not store feature names; '
                                 'a feature_spec is required.')
            feature_spec = self.tree.feature_names
        if not isinstance(feature_spec, FeatureSpec):
            feature_spec = FeatureSpec.from_column_names(list(feature_spec))
        if (self.tree.feature_names is not None
                and len(feature_spec.column_names)
                == len(self.tree.feature_names)
                and feature_spec.column_names != self.tree.feature_names):
            warnings.warn('The column names of the feature specification '
                          'differ from the feature names stored in the tree; '
                          'make sure that the columns are in the same order.',
                          UserWarning)
        self.spec = feature_spec

        if class_names is not None and not isinstance(class_names, dict):
            if len(class_names) != len(self.tree.classes):
                raise ValueError('class_names has to name every class.')
            class_names = dict(zip(self.tree.classes.tolist(), class_names))
        self.class_names = class_names
        self.paths = mf.tree_paths(self.tree, self.spec, class_names)
        self.splits = mf.path_splits(self.paths)
        self.meta_features = mf.build_meta_features(self.splits, self.spec)
        self.leaf_meta_features = mf.leaf_meta_features(self.paths,
                                                        self.meta_features)
        self._leaf_distance_matrix = None

    @property
    def classes(self):
        return self.tree.classes

    def predict_leaf(self, x):
        """Leaf id of a single (one-hot encoded) data point."""
        x = val.check_row(x, self.spec.n_columns)
        return int(self.tree.apply(x.reshape(1, -1))[0])

    def leaf_distance_matrix(self):
        """
        Leaf-to-leaf distance matrix (cached).

        Returns
        -------
        matrix : numpy.ndarray
        leaf_ids : numpy.ndarray
        leaf_classes : numpy.ndarray
        """
        if self._leaf_distance_matrix is None:
            self._leaf_distance_matrix = distances.leaf_distance_matrix(
                self.leaf_meta_features)
        lmf = self.leaf_meta_features
        return self._leaf_distance_matrix, lmf.leaf_ids, lmf.leaf_classes

    def explain_instance(self, x, instance_values=None, max_distance=None,
                         excluded=(), included=(), target_class=None,
                         counterfactual=True):
        """
        Computes the counterfactuals of a data point.

        By default these are *class-contrastive* counterfactuals -- changes
        that lead to a different prediction. With ``counterfactual=False``
        they are *class-equivalent* -- changes that move the data point to
        another leaf of the same class, i.e., that keep the prediction.

        Parameters
        ----------
        x : numpy.ndarray
            A one-hot encoded data point (1-D array or a single row).
        instance_values : dict, optional
            Base feature values reported as the current values (``'is'``);
            decoded from ``x`` with :meth:`FeatureSpec.decode` when omitted.
        max_distance : int, optional
            The largest distance to examine (default: the number of base
            features).
        excluded, included : collections.abc.Collection
            Base features that must not (only these may) appear in the
            counterfactuals; a leaf is used only if all the features that
            have to change to reach it are allowed.
        target_class : object, optional
            Explain only how to reach this class (multi-class trees); by
            default counterfactuals lead to any other class.
        counterfactual : bool
            Class-contrastive (``True``, default) or class-equivalent
            (``False``) explanations.

        Returns
        -------
        CounterfactualExplanation
        """
        x = val.check_row(x, self.spec.n_columns)
        if np.isnan(x.astype(np.float64)).any():
            raise ValueError('Data points with missing values (nan) cannot be '
                             'explained; the tree can still predict them '
                             '(predict_leaf).')
        if instance_values is None:
            instance_values = self.spec.decode(x)
        lmf = self.leaf_meta_features
        leaf = self.predict_leaf(x)
        prediction = lmf.leaf_classes[lmf.leaf_index(leaf)]

        instance = mf.instance_meta_features(x, self.meta_features)
        leaf_distances = distances.instance_leaf_distances(lmf, instance)
        support = cfs.support_set(prediction, leaf_distances, lmf.leaf_ids,
                                  lmf.leaf_classes)

        if max_distance is None:
            max_distance = self.spec.n_features
        if target_class is not None and not counterfactual:
            raise ValueError('A target class can only be given for '
                             'class-contrastive counterfactuals.')
        if target_class is not None:
            if target_class not in set(lmf.leaf_classes.tolist()):
                raise ValueError('Unknown target class: {!r}.'.format(
                    target_class))
            if target_class == prediction:
                raise ValueError('The target class has to differ from the '
                                 'prediction ({!r}).'.format(prediction))
        found = []
        for distance in range(1, max_distance + 1):
            try:
                per_leaf, per_feature = cfs.extract_counterfactuals(
                    lmf, instance, instance_values, support,
                    distance=distance, excluded=excluded, included=included,
                    target_class=target_class, counterfactual=counterfactual)
            except cfs.MissingDistance:  # no leaf this far from the point
                continue
            leaf_classes = {leaf: _python(lmf.leaf_classes[lmf.leaf_index(
                leaf)]) for leaf in per_leaf}
            ranked, sources = cfs.rank_counterfactuals(
                per_leaf, per_feature, distance, leaf_classes=leaf_classes,
                return_sources=True)
            ranked_classes = {
                rank: list(dict.fromkeys(leaf_classes[leaf]
                                         for leaf in leaves))
                for rank, leaves in sources.items()}
            found.append(DistanceCounterfactuals(
                distance, per_leaf, per_feature, ranked,
                leaf_classes=leaf_classes, ranked_classes=ranked_classes))
        return CounterfactualExplanation(
            leaf=leaf, prediction=prediction, instance_values=instance_values,
            meta_features=instance, leaf_distances=leaf_distances,
            support=support, counterfactuals=found,
            counterfactual=counterfactual)

    def describe(self, counterfactuals, integer_ranges=False):
        """
        Human-readable version of counterfactuals (see
        :func:`ctreex.counterfactuals.describe_counterfactuals`).

        Thresholds are shown in their shortest form
        (:func:`ctreex.counterfactuals.format_bound`), e.g.,
        ``'more than 0.8'`` instead of ``'more than 0.800000011920929'``.
        """
        return cfs.describe_counterfactuals(
            counterfactuals, self.spec, integer_ranges=integer_ranges)
