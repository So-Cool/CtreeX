"""
Counterfactuals of tree ensembles (experimental proof of concept).

Random forests and extra-trees (classifiers and regressors) average the
predictions of their trees, so an ensemble is described by its trees
(:class:`EnsembleStructure`, independent of scikit-learn and storable as
JSON).
Their decision regions are intersections of one leaf from every tree, so
CtreeX describes the feature space with the *cells* between the thresholds
of all the trees (the meta-features of all the trees together):
within a cell every tree -- and thus the ensemble -- predicts the same.

:class:`EnsembleCounterfactualExplainer` finds the counterfactuals of a data
point *exactly* at distances 1 and 2 by evaluating the ensemble on one point
of every cell reachable by changing one or two features:

* at distance 1, the cells of a feature that reach a target are merged into
  ranges (categories); and
* at distance 2, the pairs of cells of two features that reach a target --
  while neither change alone does -- are merged into rectangles.

The number of evaluated points grows with the square of the number of
thresholds, so ``max_evaluations`` limits it;
``included`` restricts the features to change.
Supportive explanations, contexts, audits, summaries and leaf-to-leaf
distances are not available for ensembles.
"""
import json
import os

import numpy as np

from ctreex import _validation as val
from ctreex import counterfactuals as cfs
from ctreex import meta_features as mf
from ctreex import points
from ctreex import targets as tgt
from ctreex.explainer import (CounterfactualExplanation,
                              DistanceCounterfactuals,
                              TreeCounterfactualExplainer, _python)
from ctreex.features import NUMERICAL, FeatureSpec
from ctreex.tree import CLASSIFICATION, REGRESSION, TreeStructure

__all__ = ['EnsembleStructure', 'load_ensemble',
           'EnsembleCounterfactualExplainer']

JSON_FORMAT = 'ctreex-ensemble'
JSON_FORMAT_VERSION = 1


class EnsembleStructure(object):
    """
    An ensemble averaging the predictions of its trees (random forests and
    extra-trees).

    Parameters
    ----------
    trees : list of TreeStructure
        Trees with the same features -- and classes, for classifiers.
    classes : array-like, optional
        The class labels (``None`` for regression ensembles);
        every tree predicts probabilities of these classes (its ``value``
        columns).
    feature_names : list of str, optional
    """

    def __init__(self, trees, classes=None, feature_names=None):
        trees = list(trees)
        if not trees:
            raise ValueError('An ensemble needs at least one tree.')
        self.trees = trees
        self.classes = None if classes is None else np.asarray(classes)
        self.n_features = trees[0].n_features
        self.feature_names = (None if feature_names is None
                              else [str(n) for n in feature_names])
        for tree in trees:
            if tree.n_features != self.n_features:
                raise ValueError('The trees have different numbers of '
                                 'features.')
            if (self.classes is None) != (tree.task == REGRESSION):
                raise ValueError('The trees and the ensemble have to be '
                                 'all classifiers or all regressors.')
            if self.classes is not None and \
                    tree.value.shape[1] != self.classes.shape[0]:
                raise ValueError('Every tree has to describe every class.')

    @property
    def task(self):
        return REGRESSION if self.classes is None else CLASSIFICATION

    @classmethod
    def from_sklearn(cls, model, feature_names=None):
        """
        Builds the structure from a fitted scikit-learn random forest or
        extra-trees ensemble (classifier or regressor).
        """
        estimators = getattr(model, 'estimators_', None)
        if estimators is None or not all(hasattr(e, 'tree_')
                                         for e in np.ravel(estimators)):
            raise TypeError('Expected a fitted scikit-learn random forest or '
                            'extra-trees ensemble.')
        if type(model).__name__ not in (
                'RandomForestClassifier', 'RandomForestRegressor',
                'ExtraTreesClassifier', 'ExtraTreesRegressor'):
            raise TypeError('Only random forests and extra-trees average '
                            'their trees; got {}.'.format(
                                type(model).__name__))
        if getattr(model, 'n_outputs_', 1) != 1:
            raise ValueError('Multi-output ensembles are not supported.')
        if feature_names is None and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        classes = getattr(model, 'classes_', None)
        trees = []
        for estimator in estimators:
            tree = TreeStructure.from_sklearn(estimator,
                                              feature_names=feature_names)
            if classes is not None:  # the trees encode the classes
                tree.classes = np.asarray(classes)
            trees.append(tree)
        return cls(trees, classes=classes, feature_names=feature_names)

    @classmethod
    def from_dict(cls, d):
        if d.get('format') != JSON_FORMAT:
            raise ValueError('Not a CtreeX ensemble.')
        if d.get('format_version', 0) > JSON_FORMAT_VERSION:
            raise ValueError('Unsupported ensemble format version.')
        return cls([TreeStructure.from_dict(t) for t in d['trees']],
                   classes=d.get('classes'),
                   feature_names=d.get('feature_names'))

    def to_dict(self):
        return {'format': JSON_FORMAT, 'format_version': JSON_FORMAT_VERSION,
                'task': self.task,
                'classes': (None if self.classes is None
                            else self.classes.tolist()),
                'feature_names': self.feature_names,
                'trees': [t.to_dict() for t in self.trees]}

    def to_json(self, path=None, indent=None):
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, 'w') as f:
                f.write(text)
        return text

    def _outputs(self, X):
        """Averaged class probabilities, or predictions."""
        X = val.check_array_2d(X, self.n_features).astype(np.float64)
        total = None
        for tree in self.trees:
            value = np.asarray(tree.value, dtype=np.float64)[tree.apply(X)]
            if self.classes is not None:
                value = value / value.sum(axis=1, keepdims=True)
            total = value if total is None else total + value
        return total / len(self.trees)

    def predict_proba(self, X):
        """The averaged class probabilities (classifiers only)."""
        if self.classes is None:
            raise TypeError('Regression ensembles do not predict '
                            'probabilities.')
        return self._outputs(X)

    def predict(self, X):
        """The (majority) class -- or the averaged prediction -- of each data
        point."""
        outputs = self._outputs(X)
        if self.classes is None:
            return outputs[:, 0]
        return self.classes[outputs.argmax(axis=1)]

    def __repr__(self):
        outputs = ('task=regression' if self.classes is None
                   else 'classes={}'.format(self.classes.tolist()))
        return 'EnsembleStructure(n_trees={}, n_features={}, {})'.format(
            len(self.trees), self.n_features, outputs)


def load_ensemble(path_or_object, feature_names=None):
    """
    Loads an ensemble from a JSON file (:meth:`EnsembleStructure.to_json`),
    a dictionary or a fitted scikit-learn random forest or extra-trees
    ensemble.
    """
    if isinstance(path_or_object, EnsembleStructure):
        return path_or_object
    if isinstance(path_or_object, dict):
        return EnsembleStructure.from_dict(path_or_object)
    if isinstance(path_or_object, (str, os.PathLike)):
        with open(os.fspath(path_or_object)) as f:
            return EnsembleStructure.from_dict(json.load(f))
    return EnsembleStructure.from_sklearn(path_or_object,
                                          feature_names=feature_names)


def _cell_value(feature, member):
    """A value inside a cell (``None`` if no 32-bit number lies in it)."""
    if feature.kind != NUMERICAL:
        return member.category
    if member.lower is None:
        return points.at_most(float(member.upper))
    value = points.above(float(member.lower))
    if member.upper is not None and value > float(member.upper):
        return None
    return value


def _unsupported(name):
    def method(self, *args, **kwargs):
        raise NotImplementedError(
            '{} is not available for ensembles (experimental).'.format(name))
    method.__name__ = name
    return method


def _unsupported_targets(target):
    if isinstance(target, (tgt.MinSamples, tgt.Leaves)):
        raise ValueError('{!r} refers to the leaves of a single tree; it is '
                         'not available for ensembles.'.format(target))
    for part in getattr(target, 'targets', ()):
        _unsupported_targets(part)
    if isinstance(target, tgt.Not):
        _unsupported_targets(target.target)


class EnsembleCounterfactualExplainer(object):
    """
    Exact counterfactuals of tree ensembles at distances 1 and 2
    (experimental, see :mod:`ctreex.ensemble`).

    Parameters
    ----------
    model : EnsembleStructure, scikit-learn forest, str or dict
        Anything accepted by :func:`load_ensemble`.
    feature_spec : FeatureSpec or list of str, optional
        As for :class:`ctreex.explainer.TreeCounterfactualExplainer`.
    class_names : dict or list, optional
    max_evaluations : int
        The largest number of points the ensemble is evaluated on per
        explanation.
    """

    def __init__(self, model, feature_spec=None, class_names=None,
                 max_evaluations=1000000):
        self.ensemble = load_ensemble(model)
        if feature_spec is None:
            if self.ensemble.feature_names is None:
                raise ValueError('The model does not store feature names; '
                                 'a feature_spec is required.')
            feature_spec = self.ensemble.feature_names
        if not isinstance(feature_spec, FeatureSpec):
            feature_spec = FeatureSpec.from_column_names(list(feature_spec))
        if feature_spec.n_columns != self.ensemble.n_features:
            raise ValueError('The feature specification describes {} columns '
                             'but the ensemble has {} features.'.format(
                                 feature_spec.n_columns,
                                 self.ensemble.n_features))
        self.spec = feature_spec
        self.task = self.ensemble.task
        if class_names is not None and self.task == REGRESSION:
            raise ValueError('Regression ensembles have no classes to name.')
        if class_names is not None and not isinstance(class_names, dict):
            if len(class_names) != len(self.ensemble.classes):
                raise ValueError('class_names has to name every class.')
            class_names = dict(zip(self.ensemble.classes.tolist(),
                                   class_names))
        self.class_names = class_names
        self.max_evaluations = int(max_evaluations)
        splits = {}
        for tree in self.ensemble.trees:
            for column, values in mf.path_splits(
                    mf.tree_paths(tree, self.spec)).items():
                splits.setdefault(column, set()).update(values)
        self.meta_features = mf.build_meta_features(splits, self.spec)
        self.groups = mf._feature_groups(self.meta_features)
        self._labels = None if self.task == REGRESSION else [
            class_names[c] if class_names is not None else c
            for c in self.ensemble.classes.tolist()]

    # shared with the explainer of single trees
    _check_directions = TreeCounterfactualExplainer._check_directions
    _check_ordered = TreeCounterfactualExplainer._check_ordered
    _outcome = TreeCounterfactualExplainer._outcome
    describe = TreeCounterfactualExplainer.describe
    decode = TreeCounterfactualExplainer.decode
    _change = staticmethod(TreeCounterfactualExplainer._change)

    supportive_explanations = _unsupported('supportive_explanations')
    invariance_ranges = _unsupported('invariance_ranges')
    contextualise = _unsupported('contextualise')
    feature_audit = _unsupported('feature_audit')
    monotonicity_violations = _unsupported('monotonicity_violations')
    global_summary = _unsupported('global_summary')
    leaf_distance_matrix = _unsupported('leaf_distance_matrix')
    explain_contrast = _unsupported('explain_contrast')

    def _context(self, outputs, prediction):
        """A target context whose "leaves" are evaluated points."""
        n = outputs.shape[0]
        if self.task == REGRESSION:
            return tgt.TargetContext(
                prediction=prediction, leaf=None, leaf_ids=np.arange(n),
                leaf_classes=None, leaf_probabilities=None, classes=None,
                task=REGRESSION, leaf_values=outputs[:, 0])
        labels = np.asarray(self._labels, dtype=object)
        return tgt.TargetContext(
            prediction=prediction, leaf=None, leaf_ids=np.arange(n),
            leaf_classes=labels[outputs.argmax(axis=1)],
            leaf_probabilities=outputs, classes=list(self._labels))

    def _prediction(self, x):
        output = self.ensemble._outputs(x.reshape(1, -1))[0]
        if self.task == REGRESSION:
            return float(output[0])
        return self._labels[int(output.argmax())]

    def _targets(self, target, prediction, outputs):
        context = self._context(outputs, prediction)
        mask = np.asarray(target.select(context))
        if mask.dtype != bool or mask.shape != (outputs.shape[0],):
            raise ValueError('{!r}.select has to return a boolean mask.'
                             ''.format(target))
        return mask

    def explain_instance(self, x, max_distance=2, excluded=(), included=(),
                         target=None, directions=None):
        """
        The exact counterfactuals of a data point at distances 1 and 2.

        Parameters
        ----------
        x : array-like
            One-hot encoded data point.
        max_distance : {1, 2}
        excluded, included, directions
            As for :meth:`TreeCounterfactualExplainer.explain_instance`.
        target : ctreex.targets.Target, optional
            ``OtherClass()`` by default (required for regression ensembles);
            targets referring to the leaves of a tree (``MinSamples``,
            ``Leaves``) are not available.

        Returns
        -------
        ctreex.explainer.CounterfactualExplanation
            Without leaves (``leaf``, ``leaf_distances`` and the
            ``per_leaf`` counterfactuals are empty);
            ``ranked`` holds the counterfactuals at every distance.

        Raises
        ------
        ValueError
            The explanation needs more than ``max_evaluations`` points;
            restrict the features with ``included``.
        """
        x = val.check_row(x, self.spec.n_columns).astype(np.float64)
        if np.isnan(x).any():
            raise ValueError('Data points with missing values (nan) cannot be '
                             'explained.')
        if max_distance not in (1, 2):
            raise ValueError('Ensembles are explained at distances 1 and 2.')
        excluded = val.check_features(excluded, self.spec, 'excluded')
        included = val.check_features(included, self.spec, 'included')
        directions = self._check_directions(directions)
        if target is None:
            if self.task == REGRESSION:
                raise ValueError('Regression ensembles need a target, e.g., '
                                 'ValueChange(+100).')
            target = tgt.OtherClass()
        if not isinstance(target, tgt.Target):
            raise TypeError('target has to be a ctreex.targets.Target.')
        _unsupported_targets(target)
        prediction = self._prediction(x)
        # validate the target against all the classes
        if self.task == REGRESSION:
            check = np.array([[prediction]])
        else:
            check = np.eye(len(self._labels))
        target.validate(self._context(check, prediction))

        instance_values = self.spec.decode(x)
        instance = mf.instance_meta_features(x, self.meta_features)
        options = self._options(instance, excluded, included, directions)
        sizes = [len(cells) for _, _, cells in options]
        count = sum(sizes)
        if max_distance == 2:
            count += sum(a * b for i, a in enumerate(sizes)
                         for b in sizes[i + 1:])
        if count > self.max_evaluations:
            raise ValueError(
                'The explanation needs {} evaluations (more than '
                'max_evaluations={}); restrict the features with included '
                'or explain at max_distance=1.'.format(
                    count, self.max_evaluations))

        found = []
        single = self._distance_one(x, prediction, target, options,
                                    instance_values)
        found.append(single[0])
        if max_distance == 2:
            found.append(self._distance_two(x, prediction, target, options,
                                            instance_values, single[1]))
        return CounterfactualExplanation(
            leaf=None, prediction=prediction,
            instance_values=instance_values, meta_features=instance,
            leaf_distances=None, support={},
            counterfactuals=[f for f in found if f.ranked], target=target,
            task=self.task)

    def _options(self, instance, excluded, included, directions):
        """``(feature, members, [(position, value)])``: the cells every
        allowed feature can change to."""
        options = []
        for kind, feature, indices in self.groups:
            if feature in excluded or (included and feature not in included):
                continue
            members = [self.meta_features[i] for i in indices]
            spec_feature = self.spec.features[feature]
            own = instance[indices]
            cells = []
            for position, member in enumerate(members):
                if own[position] == 1:
                    continue
                if feature in directions:
                    region = ([[member.lower, member.upper]]
                              if kind == NUMERICAL else [member.category])
                    if not cfs.in_direction(members, own, region,
                                            directions[feature]):
                        continue
                value = _cell_value(spec_feature, member)
                if value is not None:
                    cells.append((position, value))
            if cells:
                options.append((feature, members, cells))
        return options

    def _outcome_of(self, outputs):
        if self.task == REGRESSION:
            return [float(v) for v in outputs[:, 0]]
        return [self._labels[i] for i in outputs.argmax(axis=1)]

    @staticmethod
    def _region(members, positions):
        mask = np.zeros(len(members), dtype=bool)
        mask[list(positions)] = True
        if members[0].kind == NUMERICAL:
            return cfs.mask_intervals(members, mask)
        return [m.category for m, keep in zip(members, mask) if keep]

    def _rank(self, distance, entries):
        """``entries``: ``(outcome, size, counterfactual)``; ranked by size
        (the number of target cells), ties by their order."""
        order = sorted(range(len(entries)), key=lambda i: -entries[i][1])
        ranked = {r: entries[i][2] for r, i in enumerate(order)}
        outcomes = {r: entries[i][0] for r, i in enumerate(order)}
        found = DistanceCounterfactuals(distance, {}, {}, ranked)
        if self.task == REGRESSION:
            found.ranked_values = {r: (_python(o), _python(o))
                                   for r, o in outcomes.items()}
        else:
            found.ranked_classes = {r: [_python(o)]
                                    for r, o in outcomes.items()}
        return found

    def _distance_one(self, x, prediction, target, options, instance_values):
        rows, keys = [], []
        for f, (feature, _, cells) in enumerate(options):
            for position, value in cells:
                row = x.copy()
                points.set_value(row, self.spec, feature, value)
                rows.append(row)
                keys.append((f, position))
        reached = {}
        if rows:
            outputs = self.ensemble._outputs(np.array(rows))
            mask = self._targets(target, prediction, outputs)
            for (f, position), hit, outcome in zip(
                    keys, mask, self._outcome_of(outputs)):
                if hit:
                    reached[(f, position)] = outcome
        groups = {}
        for (f, position), outcome in reached.items():
            groups.setdefault((f, outcome), []).append(position)
        entries = []
        for (f, outcome), positions in groups.items():
            feature, members, _ = options[f]
            entries.append((outcome, len(positions), {feature: {
                'is': [instance_values[feature]],
                'wannabe': self._region(members, positions)}}))
        return self._rank(1, entries), set(reached)

    def _distance_two(self, x, prediction, target, options, instance_values,
                      single):
        entries = []
        for f in range(len(options)):
            for g in range(f + 1, len(options)):
                first, first_members, first_cells = options[f]
                second, second_members, second_cells = options[g]
                rows, keys = [], []
                for i, a in first_cells:
                    for j, b in second_cells:
                        # minimal changes only: neither change alone suffices
                        if (f, i) in single or (g, j) in single:
                            continue
                        row = x.copy()
                        points.set_value(row, self.spec, first, a)
                        points.set_value(row, self.spec, second, b)
                        rows.append(row)
                        keys.append((i, j))
                if not rows:
                    continue
                outputs = self.ensemble._outputs(np.array(rows))
                mask = self._targets(target, prediction, outputs)
                hits = {}
                for (i, j), hit, outcome in zip(keys, mask,
                                                self._outcome_of(outputs)):
                    if hit:
                        hits.setdefault(outcome, {}).setdefault(
                            i, set()).add(j)
                for outcome, rows_ in hits.items():
                    for firsts, seconds in self._rectangles(
                            first_members, rows_):
                        entries.append((
                            outcome, len(firsts) * len(seconds), {
                                first: {'is': [instance_values[first]],
                                        'wannabe': self._region(
                                            first_members, firsts)},
                                second: {'is': [instance_values[second]],
                                         'wannabe': self._region(
                                             second_members, seconds)}}))
        return self._rank(2, entries)

    @staticmethod
    def _rectangles(first_members, rows):
        """Merges ``{first cell: {second cells}}`` into rectangles
        ``(first cells, second cells)``: first cells with the same second
        cells are joined (only adjacent ones for numerical features)."""
        numerical = first_members[0].kind == NUMERICAL
        rectangles = []
        for i in sorted(rows):
            seconds = frozenset(rows[i])
            if rectangles and rectangles[-1][1] == seconds and (
                    not numerical or rectangles[-1][0][-1] == i - 1):
                rectangles[-1][0].append(i)
            elif not numerical and any(r[1] == seconds for r in rectangles):
                next(r for r in rectangles if r[1] == seconds)[0].append(i)
            else:
                rectangles.append(([i], seconds))
        return [(firsts, sorted(seconds)) for firsts, seconds in rectangles]

    def explanation_text(self, explanation, n=None, integer_ranges=False,
                         hide_thresholds=False, random_state=None):
        """Plain-text sentences (see
        :meth:`TreeCounterfactualExplainer.explanation_text`)."""
        return TreeCounterfactualExplainer.explanation_text(
            self, explanation, contextualised=None, n=n,
            integer_ranges=integer_ranges, hide_thresholds=hide_thresholds,
            random_state=random_state)

    def counterfactual_point(self, x, counterfactual, how='nearest'):
        """A data point satisfying a counterfactual (see
        :func:`ctreex.points.counterfactual_point`)."""
        x = val.check_row(x, self.spec.n_columns).astype(np.float64)
        return points.counterfactual_point(self.spec, x,
                                           self._change(counterfactual), how)

    def sample_counterfactual(self, x, counterfactual, n=1,
                              random_state=None):
        """Data points drawn at random from a counterfactual (see
        :func:`ctreex.points.sample_counterfactual`)."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) \
                or n < 1:
            raise ValueError('n has to be a positive integer.')
        x = val.check_row(x, self.spec.n_columns).astype(np.float64)
        return points.sample_counterfactual(
            self.spec, x, self._change(counterfactual), int(n),
            np.random.default_rng(random_state))

    def predict(self, X):
        """The predictions of the ensemble (class names applied)."""
        predictions = self.ensemble.predict(X)
        if self.class_names is None:
            return predictions
        return np.asarray([self.class_names[_python(p)]
                           for p in predictions], dtype=object)
