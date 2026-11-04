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

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex import distances
from ctreex import meta_features as mf
from ctreex import _validation as val
from ctreex.features import FeatureSpec
from ctreex.fixes import resolve_fixes
from ctreex.tree import load_tree

__all__ = ['TreeCounterfactualExplainer', 'CounterfactualExplanation',
           'DistanceCounterfactuals']

logger = logging.getLogger(__name__)


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
    """
    distance: int
    per_leaf: dict
    per_feature: dict
    ranked: dict


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
    """
    leaf: int
    prediction: object
    instance_values: dict
    meta_features: np.ndarray
    leaf_distances: np.ndarray
    support: dict
    counterfactuals: list

    @property
    def distances(self):
        return [c.distance for c in self.counterfactuals]

    def at_distance(self, distance):
        """Counterfactuals at the given distance (``None`` if absent)."""
        for c in self.counterfactuals:
            if c.distance == distance:
                return c
        return None

    def ranked(self):
        """All ranked counterfactuals as ``(distance, counterfactual)``."""
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
    implementation : {'legacy', 'fixed'}
        ``'legacy'`` reproduces the original (2018) implementation exactly;
        ``'fixed'`` enables all the behaviour-changing fixes.
    fixes : ctreex.fixes.Fixes or dict, optional
        Per-fix overrides (see :mod:`ctreex.fixes`).
    class_names : dict or list, optional
        Names of the classes (a dictionary mapping class labels to names or a
        list following ``model.classes``); used for predictions and leaves.

    Attributes
    ----------
    tree : TreeStructure
    spec : FeatureSpec
    fixes : Fixes
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

    def __init__(self, model, feature_spec=None, implementation='fixed',
                 fixes=None, class_names=None):
        self.tree = load_tree(model)
        if feature_spec is None:
            if self.tree.feature_names is None:
                raise ValueError('The model does not store feature names; '
                                 'a feature_spec is required.')
            feature_spec = self.tree.feature_names
        if not isinstance(feature_spec, FeatureSpec):
            feature_spec = FeatureSpec.from_column_names(list(feature_spec))
        self.spec = feature_spec
        self.implementation = implementation
        self.fixes = resolve_fixes(implementation, fixes)

        if class_names is not None and not isinstance(class_names, dict):
            if len(class_names) != len(self.tree.classes):
                raise ValueError('class_names has to name every class.')
            class_names = dict(zip(self.tree.classes.tolist(), class_names))
        self.class_names = class_names
        self.paths = mf.tree_paths(self.tree, self.spec, class_names)
        self.splits = mf.path_splits(self.paths)
        self.meta_features = mf.build_meta_features(
            self.splits, self.spec,
            complete_intervals=self.fixes.complete_numerical_intervals)
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
                         excluded=(), included=()):
        """
        Computes the counterfactuals of a data point.

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
            counterfactuals (see the whole_counterfactual_restrictions
            fix in :mod:`ctreex.fixes`).

        Returns
        -------
        CounterfactualExplanation
        """
        x = val.check_row(x, self.spec.n_columns)
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
        found = []
        for distance in range(1, max_distance + 1):
            try:
                per_leaf, per_feature = cfs.extract_counterfactuals(
                    lmf, instance, instance_values, support, self.spec,
                    distance=distance, excluded=excluded, included=included,
                    whole_restrictions=self.fixes.whole_counterfactual_restrictions)
            except cfs.MissingDistance:
                if self.fixes.continue_past_missing_distance:
                    continue
                break
            ranked = cfs.rank_counterfactuals(
                per_leaf, per_feature, distance,
                deterministic_order=self.fixes.deterministic_order)
            found.append(DistanceCounterfactuals(distance, per_leaf,
                                                 per_feature, ranked))
        return CounterfactualExplanation(
            leaf=leaf, prediction=prediction, instance_values=instance_values,
            meta_features=instance, leaf_distances=leaf_distances,
            support=support, counterfactuals=found)

    def describe(self, counterfactuals):
        """Human-readable version of counterfactuals (see
        :func:`ctreex.counterfactuals.describe_counterfactuals`)."""
        return cfs.describe_counterfactuals(counterfactuals, self.spec)
