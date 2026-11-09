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
3. the conditions of the *target* leaves (:mod:`ctreex.targets`) -- by
   default, leaves of another class -- at distance 1, 2, ... are turned into
   counterfactuals and ranked (:mod:`ctreex.counterfactuals`).
"""
import dataclasses
import logging
import warnings

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex import distances
from ctreex import meta_features as mf
from ctreex import _validation as val
from ctreex import audit
from ctreex import points
from ctreex import supportive as sup
from ctreex import text
from ctreex import targets as tgt
from ctreex.features import NUMERICAL, FeatureSpec
from ctreex.tree import REGRESSION, load_tree

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
        ``{leaf id: class}`` -- the class every leaf of ``per_leaf`` predicts
        (classification trees; empty for regression trees).
    ranked_classes : dict
        ``{rank: [classes]}`` -- the classes a ranked counterfactual leads to
        (counterfactuals are merged per class, so this is a single class;
        empty for regression trees).
    leaf_values : dict
        ``{leaf id: value}`` -- the prediction of every leaf of ``per_leaf``
        (regression trees; empty for classification trees).
    ranked_values : dict
        ``{rank: (smallest, largest)}`` -- the range of the predictions a
        ranked counterfactual leads to
        (counterfactuals are merged per leaf value, so the two are equal;
        empty for classification trees).
    reference_matches : dict
        ``{rank: [features]}`` -- for instance-contrastive explanations
        (:meth:`TreeCounterfactualExplainer.explain_contrast`), the changed
        features whose new values include the value of the reference data
        point; empty otherwise.
    """
    distance: int
    per_leaf: dict
    per_feature: dict
    ranked: dict
    leaf_classes: dict = dataclasses.field(default_factory=dict)
    ranked_classes: dict = dataclasses.field(default_factory=dict)
    leaf_values: dict = dataclasses.field(default_factory=dict)
    ranked_values: dict = dataclasses.field(default_factory=dict)
    reference_matches: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CounterfactualExplanation(object):
    """
    Counterfactual explanation of a single prediction.

    Attributes
    ----------
    leaf : int
        The leaf the data point falls into.
    prediction : object
        The predicted class or, for regression trees, the predicted value.
    instance_values : dict
        Base feature values of the data point.
    meta_features : numpy.ndarray
        The data point described with the meta-features.
    leaf_distances : numpy.ndarray
        Exact distances to all the leaves (in ``LeafMetaFeatures`` order).
    support : dict
        ``{distance: {'target': [...], 'non_target': [...]}}`` leaf groups
        (see :func:`ctreex.counterfactuals.support_set`).
    counterfactuals : list of DistanceCounterfactuals
        Counterfactuals ordered by increasing distance.
    target : ctreex.targets.Target
        The target the counterfactuals lead to, e.g.,
        ``OtherClass()`` (class-contrastive) or ``SameClass()``
        (class-equivalent).
    task : {'classification', 'regression'}
        The kind of the tree.
    reference : dict or None
        For instance-contrastive explanations
        (:meth:`TreeCounterfactualExplainer.explain_contrast`), the base
        feature values of the reference data point; ``None`` otherwise.
    """
    leaf: int
    prediction: object
    instance_values: dict
    meta_features: np.ndarray
    leaf_distances: np.ndarray
    support: dict
    counterfactuals: list
    target: tgt.Target = dataclasses.field(default_factory=tgt.OtherClass)
    task: str = 'classification'
    reference: dict = None

    @property
    def distances(self):
        return [c.distance for c in self.counterfactuals]

    def at_distance(self, distance):
        """Counterfactuals at the given distance (``None`` if absent)."""
        for c in self.counterfactuals:
            if c.distance == distance:
                return c
        return None

    def ranked(self, with_classes=False, with_values=False):
        """
        All ranked counterfactuals.

        Parameters
        ----------
        with_classes : bool
            Report the classes every counterfactual leads to
            (classification trees).
        with_values : bool
            Report the range of the predictions every counterfactual leads to
            (regression trees).

        Returns
        -------
        list
            ``(distance, counterfactual)`` pairs, or ``(distance, classes,
            counterfactual)`` triples with ``with_classes``, or
            ``(distance, (smallest, largest), counterfactual)`` triples with
            ``with_values``.
        """
        if with_classes and with_values:
            raise ValueError('Use either with_classes or with_values.')
        if with_classes and self.task == REGRESSION:
            raise ValueError('Regression trees have no classes; use '
                             'with_values.')
        if with_values and self.task != REGRESSION:
            raise ValueError('Classification trees have no values; use '
                             'with_classes.')
        if with_classes or with_values:
            attribute = 'ranked_classes' if with_classes else 'ranked_values'
            return [(c.distance, getattr(c, attribute)[rank], cf)
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
        The tree -- a classifier or a regressor -- i.e., anything accepted by
        :func:`ctreex.tree.load_tree`.
    feature_spec : FeatureSpec or list of str, optional
        Description of the data columns. A list of column names is converted
        with :meth:`FeatureSpec.from_column_names`; when omitted the feature
        names stored in the tree are used.
    class_names : dict or list, optional
        Names of the classes (a dictionary mapping class labels to names or a
        list following ``model.classes``); used for predictions and leaves
        (classification trees only).

    Attributes
    ----------
    tree : TreeStructure
    task : {'classification', 'regression'}
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
        self.task = self.tree.task

        if class_names is not None and self.task == REGRESSION:
            raise ValueError('Regression trees have no classes to name.')
        if class_names is not None and not isinstance(class_names, dict):
            if len(class_names) != len(self.tree.classes):
                raise ValueError('class_names has to name every class.')
            class_names = dict(zip(self.tree.classes.tolist(), class_names))
        self.class_names = class_names
        self.paths = mf.tree_paths(self.tree, self.spec, class_names)
        self.splits = mf.path_splits(self.paths)
        self.meta_features = mf.build_meta_features(self.splits, self.spec)
        self.leaf_meta_features = mf.leaf_meta_features(
            self.paths, self.meta_features, spec=self.spec)
        self._leaf_distance_matrix = None
        if self.task == REGRESSION:
            self._leaf_probabilities = self._class_labels = None
        else:
            value = np.asarray(self.tree.value, dtype=np.float64)[
                self.leaf_meta_features.leaf_ids]
            self._leaf_probabilities = value / value.sum(axis=1,
                                                         keepdims=True)
            self._class_labels = [
                class_names[c] if class_names is not None else c
                for c in self.tree.classes.tolist()]

    @property
    def classes(self):
        """The class labels (``None`` for regression trees)."""
        return self.tree.classes

    def _leaf_outputs(self):
        """The class -- or the value, for regression -- of every leaf."""
        lmf = self.leaf_meta_features
        return lmf.leaf_values if self.task == REGRESSION else lmf.leaf_classes

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
            The class of every leaf -- or, for regression trees, its value
            (pass a ``tolerance`` to the functions of
            :mod:`ctreex.counterfactuals` and :mod:`ctreex.plot` that compare
            them).
        """
        if self._leaf_distance_matrix is None:
            self._leaf_distance_matrix = distances.leaf_distance_matrix(
                self.leaf_meta_features)
        lmf = self.leaf_meta_features
        return self._leaf_distance_matrix, lmf.leaf_ids, self._leaf_outputs()

    def _check_tolerance(self, tolerance):
        if self.task == REGRESSION and tolerance is None:
            raise ValueError('Regression leaves are compared with a '
                             'tolerance: leaves whose values differ by at '
                             'most tolerance are the same (e.g., '
                             'tolerance=0 for equal values).')
        if self.task != REGRESSION and tolerance is not None:
            raise ValueError('The leaves of classification trees are '
                             'compared by their classes (no tolerance).')

    def support_set_metric(self, leaf, tolerance=None):
        """
        Groups leaves by their distance to a leaf (see
        :func:`ctreex.counterfactuals.support_set_metric`).

        Parameters
        ----------
        leaf : int
        tolerance : float, optional
            Regression trees (required): leaves whose values differ from the
            value of ``leaf`` by at most ``tolerance`` are the same.
        """
        self._check_tolerance(tolerance)
        return cfs.support_set_metric(leaf, *self.leaf_distance_matrix(),
                                      tolerance=tolerance)

    def volatility_index(self, leaf, tolerance=None):
        """
        The volatility index of a leaf (see
        :func:`ctreex.counterfactuals.volatility_index`).

        Parameters
        ----------
        leaf : int
        tolerance : float, optional
            Regression trees (required): leaves whose values differ from the
            value of ``leaf`` by at most ``tolerance`` are the same.
        """
        self._check_tolerance(tolerance)
        return cfs.volatility_index(leaf, *self.leaf_distance_matrix(),
                                    tolerance=tolerance)

    def target_context(self, leaf):
        """
        The context in which targets select leaves for a data point.

        Parameters
        ----------
        leaf : int
            The leaf of the data point (see :meth:`predict_leaf`).

        Returns
        -------
        ctreex.targets.TargetContext
        """
        lmf = self.leaf_meta_features
        return tgt.TargetContext(
            prediction=_python(self._leaf_outputs()[lmf.leaf_index(leaf)]),
            leaf=leaf, leaf_ids=lmf.leaf_ids, leaf_classes=lmf.leaf_classes,
            leaf_probabilities=self._leaf_probabilities,
            classes=(None if self._class_labels is None
                     else list(self._class_labels)),
            task=self.task, leaf_values=lmf.leaf_values,
            leaf_samples=self._leaf_statistic('n_node_samples'),
            leaf_impurity=self._leaf_statistic('impurity'))

    def _leaf_statistic(self, name):
        """A training statistic of every leaf (``None`` if not stored)."""
        values = getattr(self.tree, name)
        return None if values is None \
            else values[self.leaf_meta_features.leaf_ids]

    def _leaf_scores(self, rank_by, context):
        """``{leaf id: score}`` (higher is better) and how to aggregate the
        scores of merged counterfactuals, or ``None`` for ``frequency``."""
        if rank_by == 'frequency':
            return None, None
        if rank_by == 'samples':
            if context.leaf_samples is None:
                raise ValueError("rank_by='samples' needs the number of "
                                 'training samples of the leaves '
                                 '(n_node_samples).')
            scores = context.leaf_samples
            aggregate = sum
        elif rank_by == 'purity':
            if self.task == REGRESSION:
                if context.leaf_impurity is None:
                    raise ValueError("rank_by='purity' needs the impurity of "
                                     'the leaves of regression trees.')
                scores = -context.leaf_impurity
            else:
                scores = context.leaf_probabilities.max(axis=1)
            aggregate = min
        else:
            raise ValueError("rank_by has to be 'frequency', 'samples' or "
                             "'purity'; got {!r}.".format(rank_by))
        return ({int(leaf): float(score)
                 for leaf, score in zip(context.leaf_ids, scores)},
                aggregate)

    def _prepare(self, x, instance_values):
        """Validates a data point; returns it, its base feature values, its
        leaf and its meta-features."""
        x = val.check_row(x, self.spec.n_columns)
        if np.isnan(x.astype(np.float64)).any():
            raise ValueError('Data points with missing values (nan) cannot be '
                             'explained; the tree can still predict them '
                             '(predict_leaf).')
        if instance_values is None:
            instance_values = self.spec.decode(x)
        return (x, instance_values, self.predict_leaf(x),
                mf.instance_meta_features(x, self.meta_features))

    def _keeping_targets(self, target, leaf):
        """The targets of explanations that keep the prediction."""
        if target is None:
            if self.task == REGRESSION:
                raise ValueError('Regression trees need a target, e.g., '
                                 'ValueWithin(10).')
            target = tgt.SameClass()
        return tgt.select_targets(target, self.target_context(leaf))

    def supportive_explanations(self, x, target=None, instance_values=None):
        """
        All the smallest sets of features of a data point that guarantee its
        prediction (see :mod:`ctreex.supportive`).

        Every feature of an explanation is fixed at the meta-feature of the
        data point's value, i.e., the interval between the two thresholds
        around it or its category;
        whatever the values of the other features, the tree predicts the
        target.

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        target : ctreex.targets.Target, optional
            What the explanations guarantee;
            it has to select the leaf of the data point
            (``SameClass()`` by default, required for regression trees,
            e.g., ``ValueWithin(10)``).
        instance_values : dict, optional
            Base feature values reported as ``'is'``.

        Returns
        -------
        list of ctreex.supportive.SupportiveExplanation
        """
        x, instance_values, leaf, instance = self._prepare(x, instance_values)
        return sup.supportive_explanations(
            self.leaf_meta_features, leaf, instance, instance_values,
            self._keeping_targets(target, leaf))

    def invariance_ranges(self, x, target=None, instance_values=None):
        """
        The values of every feature that keep the prediction of a data point
        when the other features do not change (see :mod:`ctreex.supportive`).

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        target : ctreex.targets.Target, optional
            What has to be kept (``SameClass()`` by default, required for
            regression trees, e.g., ``ValueWithin(10)``).
        instance_values : dict, optional
            Base feature values reported as ``'is'``.

        Returns
        -------
        dict
            ``{feature: {'is': [value], 'wannabe': [...]}}`` for every
            feature used by the tree (``describe({0: ranges})`` works).
        """
        x, instance_values, leaf, instance = self._prepare(x, instance_values)
        return sup.invariance_ranges(
            self.leaf_meta_features, instance, instance_values,
            self._keeping_targets(target, leaf))

    def contextualise(self, explanation, max_size=None):
        """
        Adds to every ranked counterfactual of an explanation the smallest
        sets of the other features that guarantee its outcome (see
        :mod:`ctreex.supportive`).

        Every context feature is fixed at the meta-feature of the data
        point's value;
        with the change and the context, the tree predicts the target of the
        explanation whatever the values of the remaining features.

        Parameters
        ----------
        explanation : CounterfactualExplanation
        max_size : int, optional
            The largest number of context features
            (the depth of the tree by default).

        Returns
        -------
        list of ctreex.supportive.ContextualisedCounterfactual
            One per counterfactual and smallest context, ordered by distance
            and rank.

        Raises
        ------
        ValueError
            A counterfactual has no context of at most ``max_size`` features.
        """
        if max_size is None:
            max_size = max(len(p.conditions) for p in self.paths)
        targets = tgt.select_targets(explanation.target,
                                     self.target_context(explanation.leaf))
        out = []
        for found in explanation.counterfactuals:
            for rank, change in found.ranked.items():
                contexts = sup.contextualise_counterfactual(
                    self.leaf_meta_features, explanation.meta_features,
                    explanation.instance_values, targets, change, max_size)
                if not contexts:
                    raise ValueError(
                        'The counterfactual {} at distance {} has no context '
                        'of at most {} features; increase max_size.'.format(
                            rank, found.distance, max_size))
                out.extend(sup.ContextualisedCounterfactual(
                    found.distance, rank, change, context, leaves)
                    for context, leaves in contexts)
        return out

    def _check_directions(self, directions):
        """Validates ``{feature: 'increase' or 'decrease'}``."""
        if directions is None:
            return {}
        if not isinstance(directions, dict):
            raise TypeError('directions has to be a dictionary, e.g., '
                            "{'age': 'increase'}.")
        val.check_features(list(directions), self.spec, 'directions')
        for feature, direction in directions.items():
            if direction not in cfs.DIRECTIONS:
                raise ValueError("The direction of {!r} has to be 'increase' "
                                 "or 'decrease'; got {!r}.".format(
                                     feature, direction))
            self._check_ordered(feature)
        return dict(directions)

    def _check_ordered(self, feature):
        spec_feature = self.spec.features[feature]
        if spec_feature.kind != NUMERICAL and not spec_feature.ordered:
            raise ValueError('{!r} is an unordered categorical feature; '
                             'directions need numerical or ordered features '
                             '(Feature(..., ordered=True)).'.format(feature))

    def monotonicity_violations(self, feature, direction='increase',
                                positive=None):
        """
        Pairs of leaves where changing a feature in a direction, with the
        other features unchanged, makes the prediction worse
        (see :func:`ctreex.audit.monotonicity_violations`).

        Parameters
        ----------
        feature : str
            A numerical or ordered categorical feature.
        direction : {'increase', 'decrease'}
            The direction in which changing the feature should never make the
            prediction worse.
        positive : object, optional
            For classification trees (required), the better class, leaving
            which makes the prediction worse;
            for regression trees, a smaller prediction is worse.

        Returns
        -------
        list of ctreex.audit.MonotonicityViolation
        """
        val.check_features([feature], self.spec, 'feature')
        self._check_ordered(feature)
        if direction not in cfs.DIRECTIONS:
            raise ValueError("direction has to be 'increase' or 'decrease'; "
                             'got {!r}.'.format(direction))
        if self.task == REGRESSION:
            if positive is not None:
                raise ValueError('Regression trees have no positive class; '
                                 'a smaller prediction is worse.')

            def worse(before, after):
                return after < before
        else:
            if positive not in set(self.leaf_meta_features.leaf_classes
                                   .tolist()):
                raise ValueError('positive has to be a class predicted by '
                                 'some leaf; got {!r}.'.format(positive))

            def worse(before, after):
                return before == positive and after != positive
        outputs = [_python(o) for o in self._leaf_outputs()]
        return audit.monotonicity_violations(
            self.leaf_meta_features, feature, direction, outputs, worse)

    def feature_audit(self, feature, X=None, target=None, tolerance=None):
        """
        Where a feature alone changes the prediction (see
        :func:`ctreex.audit.feature_flips`), e.g., to check whether a
        protected attribute decides the outcome for some data points.

        Parameters
        ----------
        feature : str
        X : array-like, optional
            One-hot encoded data points;
            when given, the audit also lists the data points whose prediction
            the feature can change (``alone`` and ``closest``).
        target : ctreex.targets.Target, optional
            The target of the counterfactuals of the data points
            (``OtherClass()`` by default, required for regression trees with
            data, e.g., ``ValueChange(+100)``).
        tolerance : float, optional
            Regression trees (required): the predictions of two leaves are
            different when they differ by more than the tolerance.

        Returns
        -------
        ctreex.audit.FeatureAudit
        """
        val.check_features([feature], self.spec, 'feature')
        if self.task == REGRESSION:
            if tolerance is None:
                raise ValueError('Regression leaves are compared with a '
                                 'tolerance: predictions that differ by more '
                                 'than tolerance are different.')
            if tolerance < 0:
                raise ValueError('tolerance cannot be negative.')

            def changed(before, after):
                return abs(after - before) > tolerance
        else:
            if tolerance is not None:
                raise ValueError('The leaves of classification trees are '
                                 'compared by their classes (no tolerance).')

            def changed(before, after):
                return before != after
        outputs = [_python(o) for o in self._leaf_outputs()]
        flips = audit.feature_flips(self.leaf_meta_features, feature,
                                    outputs, changed)
        affected = set(f.from_leaf for f in flips)
        ids = self.leaf_meta_features.leaf_ids.tolist()
        result = audit.FeatureAudit(
            feature=feature, flips=flips,
            affected_leaves=[leaf for leaf in ids if leaf in affected],
            unaffected_leaves=[leaf for leaf in ids if leaf not in affected])
        if X is None:
            return result
        X = val.check_array_2d(X, self.spec.n_columns)
        result.n_points, result.alone, result.closest = X.shape[0], [], []
        for i, x in enumerate(X):
            explanation = self.explain_instance(x, target=target)
            found = [c for c in explanation.counterfactuals if c.per_leaf]
            if not found:
                continue
            if found[0].distance == 1 and feature in found[0].per_feature:
                result.alone.append(i)
            if feature in found[0].per_feature:
                result.closest.append(i)
        return result

    def explain_contrast(self, x, x_reference, same_leaf=False,
                         tolerance=None, **kwargs):
        """
        Explains how a data point would have to change to be treated like a
        reference data point (instance-contrastive explanations).

        The counterfactuals of ``x`` lead to the leaves predicting the outcome
        of ``x_reference`` -- or, with ``same_leaf``, to its leaf -- and at
        every distance those whose new values include the values of the
        reference come first (``DistanceCounterfactuals.reference_matches``
        lists these features).

        Parameters
        ----------
        x, x_reference : array-like
            One-hot encoded data points.
        same_leaf : bool
            Lead to the leaf of the reference instead of its outcome;
            then the two data points may have the same prediction, but not
            the same leaf.
        tolerance : float, optional
            Regression trees (required unless ``same_leaf``): the leaves
            predicting at most ``tolerance`` more or less than the reference
            are targets;
            the prediction of ``x`` has to differ by more.
        **kwargs
            Passed to :meth:`explain_instance` (e.g., ``max_distance``,
            ``excluded``, ``included``, ``directions``, ``rank_by``), except
            ``target``.

        Returns
        -------
        CounterfactualExplanation
            With the ``reference`` values.

        Raises
        ------
        ValueError
            The two data points have the same outcome (and ``same_leaf`` is
            not set) or the same leaf.
        """
        if 'target' in kwargs:
            raise TypeError('explain_contrast sets the target from the '
                            'reference data point.')
        x_reference, reference_values, reference_leaf, reference_meta = \
            self._prepare(x_reference, None)
        leaf = self.predict_leaf(val.check_row(x, self.spec.n_columns))
        context = self.target_context(reference_leaf)
        own = self.target_context(leaf).prediction
        if same_leaf:
            if leaf == reference_leaf:
                raise ValueError('The data points share their leaf.')
            if tolerance is not None:
                raise ValueError('tolerance is not used with same_leaf.')
            target = tgt.Leaves(reference_leaf)
        elif self.task == REGRESSION:
            if tolerance is None or tolerance < 0:
                raise ValueError('Regression trees need a non-negative '
                                 'tolerance: the leaves predicting at most '
                                 'tolerance more or less than the reference '
                                 'are targets.')
            if abs(own - context.prediction) <= tolerance:
                raise ValueError('The predictions of the data points differ '
                                 'by at most the tolerance; use '
                                 'same_leaf=True to reach the leaf of the '
                                 'reference.')
            target = (tgt.ValueAtLeast(context.prediction - tolerance)
                      & tgt.ValueAtMost(context.prediction + tolerance))
        else:
            if tolerance is not None:
                raise ValueError('The leaves of classification trees are '
                                 'compared by their classes (no tolerance).')
            if own == context.prediction:
                raise ValueError('The data points have the same prediction '
                                 '({!r}); use same_leaf=True to reach the '
                                 'leaf of the reference.'.format(own))
            target = tgt.TargetClass(context.prediction)
        explanation = self.explain_instance(x, target=target, **kwargs)
        lmf = self.leaf_meta_features
        for found in explanation.counterfactuals:
            matches = {rank: sup.matching_features(lmf, reference_meta, cf)
                       for rank, cf in found.ranked.items()}
            # more matching features first; ties keep their order
            order = sorted(found.ranked, key=lambda r: -len(matches[r]))
            found.ranked = {new: found.ranked[old]
                            for new, old in enumerate(order)}
            for name in ('ranked_classes', 'ranked_values'):
                ranked = getattr(found, name)
                if ranked:
                    setattr(found, name, {new: ranked[old] for new, old
                                          in enumerate(order)})
            found.reference_matches = {new: matches[old]
                                       for new, old in enumerate(order)}
        explanation.reference = reference_values
        return explanation

    def explain_instance(self, x, instance_values=None, max_distance=None,
                         excluded=(), included=(), target=None,
                         rank_by='frequency', directions=None):
        """
        Computes the counterfactuals of a data point.

        The counterfactuals are the changes that move the data point to the
        closest *target* leaves (see :mod:`ctreex.targets`).
        By default (``OtherClass()``) these are *class-contrastive*
        counterfactuals -- changes that lead to a different prediction;
        ``TargetClass(label, ...)`` leads to given classes (multi-class
        trees);
        ``SameClass()`` yields *class-equivalent* explanations -- changes
        that move the data point to another leaf of the same class, i.e.,
        that keep the prediction.
        Regression trees need a target, e.g., ``ValueChange(+100)`` -- the
        changes that increase the prediction by at least 100.

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
            counterfactuals;
            a leaf is used only if all the features that have to change to
            reach it are allowed.
            The names have to be features of the specification
            (``FeatureSpec.features``);
            features the tree does not use have no effect.
        target : ctreex.targets.Target, optional
            Selects the leaves the counterfactuals lead to
            (default: ``OtherClass()``; required for regression trees).
        rank_by : {'frequency', 'samples', 'purity'}
            Orders the counterfactuals at every distance:
            ``'frequency'`` (default) by how many target leaves need their
            features to change;
            ``'samples'`` by the number of training samples of their leaves
            (the total of the merged leaves, needs ``n_node_samples``); and
            ``'purity'`` by the purity of their leaves -- the proportion of
            the majority class or the impurity of regression leaves (lower
            first, needs ``impurity``) -- taking the least pure of the merged
            leaves.
            Ties keep the ``'frequency'`` order.
        directions : dict, optional
            ``{feature: 'increase' or 'decrease'}`` -- numerical or ordered
            categorical (``Feature.ordered``) features that may only change
            in the given direction, e.g., ``{'age': 'increase'}``;
            values in the other direction are left out and leaves reachable
            only that way are not used.

        Returns
        -------
        CounterfactualExplanation
        """
        x = val.check_row(x, self.spec.n_columns)
        excluded = val.check_features(excluded, self.spec, 'excluded')
        included = val.check_features(included, self.spec, 'included')
        directions = self._check_directions(directions)
        if np.isnan(x.astype(np.float64)).any():
            raise ValueError('Data points with missing values (nan) cannot be '
                             'explained; the tree can still predict them '
                             '(predict_leaf).')
        if instance_values is None:
            instance_values = self.spec.decode(x)
        if target is None:
            if self.task == REGRESSION:
                raise ValueError('Regression trees need a target, e.g., '
                                 'ValueChange(+100) or ValueAtMost(50).')
            target = tgt.OtherClass()
        lmf = self.leaf_meta_features
        leaf = self.predict_leaf(x)
        context = self.target_context(leaf)
        prediction = context.prediction
        targets = tgt.select_targets(target, context)
        leaf_scores, aggregate = self._leaf_scores(rank_by, context)

        instance = mf.instance_meta_features(x, self.meta_features)
        leaf_distances = distances.instance_leaf_distances(lmf, instance)
        support = cfs.support_set(leaf_distances, lmf.leaf_ids, targets)

        if max_distance is None:
            max_distance = self.spec.n_features
        found = []
        for distance in range(1, max_distance + 1):
            try:
                per_leaf, per_feature = cfs.extract_counterfactuals(
                    lmf, instance, instance_values, support,
                    distance=distance, excluded=excluded, included=included,
                    directions=directions)
            except cfs.MissingDistance:  # no leaf this far from the point
                continue
            outputs = {leaf: _python(self._leaf_outputs()[lmf.leaf_index(
                leaf)]) for leaf in per_leaf}
            # merged and scored per class (per value for regression trees)
            ranked, sources = cfs.rank_counterfactuals(
                per_leaf, per_feature, distance, leaf_classes=outputs,
                return_sources=True, leaf_scores=leaf_scores,
                aggregate=aggregate)
            if self.task == REGRESSION:
                reported = {'leaf_values': outputs, 'ranked_values': {
                    rank: (min(outputs[leaf] for leaf in leaves),
                           max(outputs[leaf] for leaf in leaves))
                    for rank, leaves in sources.items()}}
            else:
                reported = {'leaf_classes': outputs, 'ranked_classes': {
                    rank: list(dict.fromkeys(outputs[leaf]
                                             for leaf in leaves))
                    for rank, leaves in sources.items()}}
            found.append(DistanceCounterfactuals(
                distance, per_leaf, per_feature, ranked, **reported))
        return CounterfactualExplanation(
            leaf=leaf, prediction=prediction, instance_values=instance_values,
            meta_features=instance, leaf_distances=leaf_distances,
            support=support, counterfactuals=found, target=target,
            task=self.task)

    def describe(self, counterfactuals, integer_ranges=False,
                 hide_thresholds=False, random_state=None):
        """
        Human-readable version of counterfactuals (see
        :func:`ctreex.counterfactuals.describe_counterfactuals`).

        Thresholds are shown in their shortest form
        (:func:`ctreex.counterfactuals.format_bound`), e.g.,
        ``'more than 0.8'`` instead of ``'more than 0.800000011920929'``.

        Parameters
        ----------
        counterfactuals : dict
            ``{key: {feature: {'is': [...], 'wannabe': [...]}}}``.
        integer_ranges : bool
            Describe the ranges of integer features with whole numbers.
        hide_thresholds : bool
            Replace the ranges of numerical features with a value drawn at
            random from them (:func:`ctreex.points.sample_value`), so the
            description does not reveal the thresholds of the tree;
            unbounded ranges need the minimum and maximum of the features
            (``FeatureSpec.fit_ranges``).
        random_state : int or numpy.random.Generator, optional
            The seed (generator) of ``hide_thresholds``.
        """
        if hide_thresholds:
            rng = np.random.default_rng(random_state)
            hidden = {}
            for key, changes in counterfactuals.items():
                hidden[key] = {}
                for name, part in changes.items():
                    feature = self.spec.features[name]
                    if feature.kind == NUMERICAL:
                        value = points.sample_value(feature, part['wannabe'],
                                                    rng)
                        hidden[key][feature.display_name] = {
                            'is': list(part['is']), 'wannabe': [value]}
                    else:
                        hidden[key][feature.display_name] = cfs\
                            .describe_counterfactuals(
                                {0: {name: part}}, self.spec)[0][
                                    feature.display_name]
            return hidden
        return cfs.describe_counterfactuals(
            counterfactuals, self.spec, integer_ranges=integer_ranges)

    def _outcome(self, explanation, found, rank):
        """The outcome clause of a ranked counterfactual."""
        prediction = text.format_value(explanation.prediction)
        if self.task == REGRESSION:
            low, high = found.ranked_values[rank]
            value = text.format_value(low) if low == high else \
                'between {} and {}'.format(text.format_value(low),
                                           text.format_value(high))
            return 'the prediction would have been {} (instead of {})'\
                ''.format(value, prediction)
        classes = found.ranked_classes[rank]
        if classes == [explanation.prediction]:
            return 'the prediction would still have been {}'.format(
                prediction)
        return 'the prediction would have been {}'.format(
            text.join([text.format_value(c) for c in classes], 'or'))

    def explanation_text(self, explanation, contextualised=None, n=None,
                         integer_ranges=False, hide_thresholds=False,
                         random_state=None):
        """
        A counterfactual explanation as plain-text sentences from a generic
        template (see :mod:`ctreex.text`), e.g.,
        "Had duration been less or equal to 22.5 (instead of 48), the
        prediction would have been low risk."

        Parameters
        ----------
        explanation : CounterfactualExplanation
        contextualised : list, optional
            The output of :meth:`contextualise`
            (``ctreex.supportive.ContextualisedCounterfactual``);
            when given, every counterfactual ends with its context
            ("..., provided that ...").
        n : int, optional
            The largest number of counterfactual sentences (all by default).
        integer_ranges, hide_thresholds, random_state
            See :meth:`describe`.

        Returns
        -------
        list of str
            The prediction followed by one sentence per (contextualised)
            counterfactual, by distance and rank.
        """
        def described(changes):
            return self.describe({0: changes}, integer_ranges=integer_ranges,
                                 hide_thresholds=hide_thresholds,
                                 random_state=rng)[0]
        rng = np.random.default_rng(random_state)
        if contextualised is None:
            items = [(found, rank, cf, None)
                     for found in explanation.counterfactuals
                     for rank, cf in found.ranked.items()]
        else:
            items = [(explanation.at_distance(c.distance), c.rank, c.change,
                      c.context) for c in contextualised]
        if n is not None:
            items = items[:n]
        sentences = [text.prediction_sentence(explanation.prediction)]
        for found, rank, change, context in items:
            sentences.append(text.counterfactual_sentence(
                described(change), self._outcome(explanation, found, rank),
                described(context) if context else None))
        return sentences

    def supportive_text(self, x, target=None, integer_ranges=False):
        """
        The supportive explanations of a data point
        (:meth:`supportive_explanations`) as plain-text sentences, e.g.,
        "The prediction is low risk since property is real estate and
        duration is less or equal to 11.5, whatever the other features."

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        target : ctreex.targets.Target, optional
            See :meth:`supportive_explanations`.
        integer_ranges : bool
            See :meth:`describe`.

        Returns
        -------
        list of str
            One sentence per explanation.
        """
        found = self.supportive_explanations(x, target=target)
        prediction = text.format_value(
            self.target_context(self.predict_leaf(x)).prediction)
        if isinstance(target, tgt.ValueWithin):
            prediction = 'within {} of {}'.format(
                text.format_value(target.tolerance), prediction)
        elif target is not None and not isinstance(target, tgt.SameClass):
            prediction = '{} ({!r} holds)'.format(prediction, target)
        sentences = []
        for explanation in found:
            conditions = self.describe({0: explanation.conditions},
                                       integer_ranges=integer_ranges)[0]
            sentences.append(text.supportive_sentence(conditions, prediction))
        return sentences

    def decode(self, X):
        """
        Base feature values of one-hot encoded data points (see
        :meth:`ctreex.features.FeatureSpec.decode`).

        Returns
        -------
        dict or list of dict
            One dictionary for a single data point, a list for a 2-D array.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return self.spec.decode(X)
        return [self.spec.decode(row) for row in X]

    @staticmethod
    def _change(counterfactual):
        """The changes of a counterfactual or a contextualised one."""
        if isinstance(counterfactual, sup.ContextualisedCounterfactual):
            return counterfactual.change
        if not isinstance(counterfactual, dict):
            raise TypeError('Expected a counterfactual '
                            "({{feature: {{'is': ..., 'wannabe': ...}}}}) or "
                            'a ContextualisedCounterfactual; got {!r}.'.format(
                                counterfactual))
        return counterfactual

    def counterfactual_leaves(self, x, counterfactual):
        """
        The leaves a counterfactual of a data point leads to, i.e., the leaves
        consistent with the changed features in their ranges and the other
        features at the values of the data point.

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        counterfactual : dict or ctreex.supportive.ContextualisedCounterfactual

        Returns
        -------
        list of int
        """
        x, _, _, instance = self._prepare(x, None)
        mask = sup.counterfactual_leaves(self.leaf_meta_features, instance,
                                         self._change(counterfactual))
        return self.leaf_meta_features.leaf_ids[mask].tolist()

    def _check_points(self, x, counterfactual, rows):
        """Every data point has to reach a leaf of the counterfactual."""
        expected = set(self.counterfactual_leaves(x, counterfactual))
        reached = self.tree.apply(rows)
        if not set(reached.tolist()) <= expected:  # pragma: no cover
            raise RuntimeError('Internal error: a data point of the '
                               'counterfactual reaches leaf {} instead of '
                               '{}.'.format(sorted(set(reached.tolist()) -
                                                   expected),
                                            sorted(expected)))

    def counterfactual_point(self, x, counterfactual, how='nearest'):
        """
        A data point satisfying a counterfactual (see
        :func:`ctreex.points.counterfactual_point`).

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        counterfactual : dict or ctreex.supportive.ContextualisedCounterfactual
            E.g., ``explanation.at_distance(1).ranked[0]``.
        how : {'nearest', 'midpoint'}
            The values of the changed features nearest to the data point or
            in the middle of their ranges (unbounded ranges need the minimum
            and maximum of the features);
            the other features keep the values of the data point.

        Returns
        -------
        numpy.ndarray
            One-hot encoded data point (see :meth:`decode`).
        """
        x, _, _, _ = self._prepare(x, None)
        row = points.counterfactual_point(self.spec, x,
                                          self._change(counterfactual), how)
        self._check_points(x, counterfactual, row.reshape(1, -1))
        return row

    def sample_counterfactual(self, x, counterfactual, n=1,
                              random_state=None):
        """
        Data points drawn at random from a counterfactual (see
        :func:`ctreex.points.sample_counterfactual`), e.g., to share a
        counterfactual without revealing the thresholds of the tree.

        Every changed feature is drawn uniformly from its ranges (unbounded
        ranges need the minimum and maximum of the features, see
        ``FeatureSpec.fit_ranges``) or categories;
        the other features keep the values of the data point.

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        counterfactual : dict or ctreex.supportive.ContextualisedCounterfactual
            E.g., ``explanation.at_distance(1).ranked[0]``.
        n : int
            The number of data points.
        random_state : int or numpy.random.Generator, optional
            The seed (generator), for reproducible samples.

        Returns
        -------
        numpy.ndarray
            ``(n, n_columns)`` one-hot encoded data points
            (see :meth:`decode`).
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) \
                or n < 1:
            raise ValueError('n has to be a positive integer.')
        x, _, _, _ = self._prepare(x, None)
        rows = points.sample_counterfactual(
            self.spec, x, self._change(counterfactual), int(n),
            np.random.default_rng(random_state))
        self._check_points(x, counterfactual, rows)
        return rows

    def exemplars(self, X, leaves, n=None, random_state=None):
        """
        Data points (e.g., training data) that fall into given leaves.

        Exemplars illustrate a leaf -- e.g., the leaves of a counterfactual
        (:meth:`counterfactual_leaves`) -- with real data points;
        beware that they reveal these data points.

        Parameters
        ----------
        X : array-like
            One-hot encoded data points.
        leaves : int or list of int
        n : int, optional
            Draw at most ``n`` of them at random (all by default).
        random_state : int or numpy.random.Generator, optional

        Returns
        -------
        numpy.ndarray
            The data points (rows of ``X``, in their order).
        """
        X = val.check_array_2d(X, self.spec.n_columns)
        leaves = [leaves] if np.ndim(leaves) == 0 else list(leaves)
        known = set(self.leaf_meta_features.leaf_ids.tolist())
        unknown = [leaf for leaf in leaves if leaf not in known]
        if unknown:
            raise ValueError('These nodes are not leaves: {}.'.format(
                ', '.join(map(str, unknown))))
        indices = np.flatnonzero(np.isin(self.tree.apply(X), leaves))
        if n is not None and n < indices.size:
            rng = np.random.default_rng(random_state)
            indices = np.sort(rng.choice(indices, size=n, replace=False))
        return X[indices]
