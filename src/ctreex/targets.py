"""
Targets: which leaves the explanations of a data point lead to.

The explainer finds the leaves closest to a data point among the *target*
leaves and turns their conditions into explanations.
A target selects these leaves.
Classification trees:

* :class:`OtherClass` (the default) -- leaves of any class other than the
  prediction, i.e., *class-contrastive* counterfactuals;
* :class:`TargetClass` -- leaves of one or more given classes (multi-class
  trees);
* :class:`SameClass` -- other leaves of the predicted class, i.e.,
  *class-equivalent* explanations:
  the changes that keep the prediction; and
* :class:`ClassProbability` -- leaves of given classes whose majority is at
  least a given proportion (confident leaves).

Regression trees (a target has to be given):

* :class:`ValueChange` -- leaves whose prediction is larger (smaller) than
  the prediction of the data point by at least a given amount;
* :class:`ValueAtLeast` and :class:`ValueAtMost` -- leaves whose prediction
  is at least (at most) a given value; and
* :class:`ValueWithin` -- leaves whose prediction differs from the prediction
  of the data point by at most a given tolerance, i.e., the changes that
  (approximately) keep the prediction.

Any tree:

* :class:`MinSamples` -- leaves with at least a given number of training
  samples (trees that store them, e.g., built from scikit-learn).

Targets combine with ``&`` for both, ``|`` for either and ``~`` for not;
e.g., ``OtherClass() & MinSamples(20)`` leads to leaves of another class
backed by at least 20 training samples
(see :class:`AllOf`, :class:`AnyOf` and :class:`Not`).

Custom targets subclass :class:`Target` and implement :meth:`Target.select`,
which receives a :class:`TargetContext` describing the prediction and all
the leaves and returns a boolean mask over the leaves;
e.g., the leaves of the predicted class whose majority is at least 90%::

    class ConfidentSameClass(Target):
        def select(self, context):
            same = context.leaf_classes == context.prediction
            return same & (context.leaf_probabilities.max(axis=1) >= 0.9)

    explainer.explain_instance(x, target=ConfidentSameClass())

The leaf of the data point itself is never used (it is at distance ``0``).
"""
import dataclasses
import math

import numpy as np

from ctreex.tree import CLASSIFICATION, REGRESSION

__all__ = ['TargetContext', 'Target', 'OtherClass', 'TargetClass',
           'SameClass', 'ClassProbability', 'ValueChange', 'ValueAtLeast',
           'ValueAtMost', 'ValueWithin', 'MinSamples', 'AllOf', 'AnyOf',
           'Not', 'select_targets']

_CLASSIFICATION_TARGETS = 'OtherClass, TargetClass, SameClass or ' \
    'ClassProbability'
_REGRESSION_TARGETS = 'ValueChange, ValueAtLeast, ValueAtMost or ValueWithin'


@dataclasses.dataclass(frozen=True)
class TargetContext(object):
    """
    The prediction of a data point and the leaves of the tree.

    The class attributes are ``None`` for regression trees and
    ``leaf_values`` is ``None`` for classification trees.

    Attributes
    ----------
    prediction : object
        The predicted class (its name when the explainer has class names)
        or, for regression trees, the predicted value.
    leaf : int
        The leaf the data point falls into.
    leaf_ids : numpy.ndarray
        All the leaf ids
        (in the order of ``TreeCounterfactualExplainer.leaf_meta_features``).
    leaf_classes : numpy.ndarray or None
        The (majority) class of every leaf.
    leaf_probabilities : numpy.ndarray or None
        ``(n_leaves, n_classes)`` class proportions of every leaf.
    classes : list or None
        The classes of the columns of ``leaf_probabilities``.
    task : {'classification', 'regression'}
        The kind of the tree.
    leaf_values : numpy.ndarray or None
        The prediction of every leaf (regression trees).
    leaf_samples : numpy.ndarray or None
        The number of training samples in every leaf
        (``None`` when the tree does not store it).
    leaf_impurity : numpy.ndarray or None
        The impurity of every leaf
        (``None`` when the tree does not store it).
    """
    prediction: object
    leaf: int
    leaf_ids: np.ndarray
    leaf_classes: np.ndarray
    leaf_probabilities: np.ndarray
    classes: list
    task: str = CLASSIFICATION
    leaf_values: np.ndarray = None
    leaf_samples: np.ndarray = None
    leaf_impurity: np.ndarray = None


def _require(target, context, task):
    """Checks that a target is used with its kind of tree."""
    if context.task != task:
        alternatives = (_REGRESSION_TARGETS if context.task == REGRESSION
                        else _CLASSIFICATION_TARGETS)
        raise ValueError('{!r} is a target of {} trees; {} trees use {}.'
                         ''.format(target, task, context.task, alternatives))


def _check_labels(target, labels, context):
    """Checks that some leaf predicts every class."""
    known = set(context.leaf_classes.tolist())
    unknown = [label for label in labels if label not in known]
    if unknown:
        raise ValueError('Unknown classes of {!r} (no leaf predicts them): '
                         '{}.'.format(target, ', '.join(map(repr, unknown))))


def _labels(name, labels):
    if not labels:
        raise TypeError('{} needs at least one class.'.format(name))
    return tuple(dict.fromkeys(labels))


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         np.number)) \
            or not math.isfinite(value):
        raise ValueError('{} has to be a finite number; got {!r}.'.format(
            name, value))
    return value


class Target(object):
    """
    Base class of targets.

    Subclasses implement :meth:`select` and may implement :meth:`validate`.
    Targets combine with ``&`` (:class:`AllOf`), ``|`` (:class:`AnyOf`) and
    ``~`` (:class:`Not`).
    """

    def __and__(self, other):
        return AllOf(self, other)

    def __or__(self, other):
        return AnyOf(self, other)

    def __invert__(self):
        return Not(self)

    def validate(self, context):
        """
        Checks that the target can be used for a prediction.

        Raises an exception (e.g., ``ValueError``) when it cannot;
        does nothing by default.

        Parameters
        ----------
        context : TargetContext
        """

    def select(self, context):
        """
        Selects the target leaves.

        Parameters
        ----------
        context : TargetContext

        Returns
        -------
        numpy.ndarray
            Boolean mask over ``context.leaf_ids``.
        """
        raise NotImplementedError('Targets have to implement select.')


@dataclasses.dataclass(frozen=True)
class OtherClass(Target):
    """Leaves of any class other than the prediction (class-contrastive)."""

    def validate(self, context):
        _require(self, context, CLASSIFICATION)

    def select(self, context):
        return context.leaf_classes != context.prediction


@dataclasses.dataclass(frozen=True, init=False)
class TargetClass(Target):
    """
    Leaves of one or more given classes, which differ from the prediction.

    ``TargetClass('a')`` leads to class ``a`` and ``TargetClass('a', 'b')``
    to class ``a`` or ``b``.

    Attributes
    ----------
    labels : tuple
        The classes (their names when the explainer has class names).
    """
    labels: tuple

    def __init__(self, *labels):
        object.__setattr__(self, 'labels', _labels('TargetClass', labels))

    def validate(self, context):
        _require(self, context, CLASSIFICATION)
        _check_labels(self, self.labels, context)
        if context.prediction in self.labels:
            raise ValueError('The target classes have to differ from the '
                             'prediction ({!r}); use SameClass() for '
                             'class-equivalent explanations.'.format(
                                 context.prediction))

    def select(self, context):
        return np.isin(context.leaf_classes, self.labels)


@dataclasses.dataclass(frozen=True)
class SameClass(Target):
    """Other leaves of the predicted class (class-equivalent)."""

    def validate(self, context):
        _require(self, context, CLASSIFICATION)

    def select(self, context):
        return context.leaf_classes == context.prediction


@dataclasses.dataclass(frozen=True, init=False)
class ClassProbability(Target):
    """
    Leaves of given classes whose majority is at least a given proportion.

    A leaf is a target when it predicts one of the classes and the proportion
    of that class in the leaf is at least ``at_least``;
    e.g., ``ClassProbability('good', at_least=0.8)`` leads to leaves that
    predict ``good`` with at least 80% of their training data.
    The classes may include the prediction (e.g., to explain how to be
    classified the same way with more confidence).

    Attributes
    ----------
    labels : tuple
        The classes (their names when the explainer has class names).
    at_least : float
        The smallest proportion of the class in a target leaf
        (between 0 and 1).
    """
    labels: tuple
    at_least: float

    def __init__(self, *labels, at_least):
        at_least = _number('at_least', at_least)
        if not 0 <= at_least <= 1:
            raise ValueError('at_least has to be between 0 and 1; got '
                             '{!r}.'.format(at_least))
        object.__setattr__(self, 'labels', _labels('ClassProbability',
                                                   labels))
        object.__setattr__(self, 'at_least', at_least)

    def validate(self, context):
        _require(self, context, CLASSIFICATION)
        _check_labels(self, self.labels, context)

    def select(self, context):
        # the class of a leaf is its majority class
        confident = context.leaf_probabilities.max(axis=1) >= self.at_least
        return np.isin(context.leaf_classes, self.labels) & confident


@dataclasses.dataclass(frozen=True)
class ValueChange(Target):
    """
    Leaves whose prediction differs from the prediction by at least ``delta``.

    For a positive ``delta`` the target leaves predict at least
    ``prediction + delta`` and for a negative one at most
    ``prediction + delta``;
    e.g., ``ValueChange(+100)`` leads to leaves predicting at least 100 more.

    Attributes
    ----------
    delta : float
        The smallest change of the prediction (non-zero).
    """
    delta: float

    def __post_init__(self):
        if _number('delta', self.delta) == 0:
            raise ValueError('delta has to be positive or negative; use '
                             'ValueWithin for the changes that keep the '
                             'prediction.')

    def validate(self, context):
        _require(self, context, REGRESSION)

    def select(self, context):
        bound = context.prediction + self.delta
        if self.delta > 0:
            return context.leaf_values >= bound
        return context.leaf_values <= bound


@dataclasses.dataclass(frozen=True)
class ValueAtLeast(Target):
    """
    Leaves predicting at least ``value``.

    Attributes
    ----------
    value : float
    """
    value: float

    def __post_init__(self):
        _number('value', self.value)

    def validate(self, context):
        _require(self, context, REGRESSION)

    def select(self, context):
        return context.leaf_values >= self.value


@dataclasses.dataclass(frozen=True)
class ValueAtMost(Target):
    """
    Leaves predicting at most ``value``.

    Attributes
    ----------
    value : float
    """
    value: float

    def __post_init__(self):
        _number('value', self.value)

    def validate(self, context):
        _require(self, context, REGRESSION)

    def select(self, context):
        return context.leaf_values <= self.value


@dataclasses.dataclass(frozen=True)
class ValueWithin(Target):
    """
    Leaves whose prediction differs from the prediction by at most
    ``tolerance``, i.e., the changes that (approximately) keep the prediction.

    Attributes
    ----------
    tolerance : float
        The largest change of the prediction (non-negative).
    """
    tolerance: float

    def __post_init__(self):
        if _number('tolerance', self.tolerance) < 0:
            raise ValueError('tolerance cannot be negative.')

    def validate(self, context):
        _require(self, context, REGRESSION)

    def select(self, context):
        return np.abs(context.leaf_values - context.prediction) \
            <= self.tolerance


@dataclasses.dataclass(frozen=True)
class MinSamples(Target):
    """
    Leaves with at least ``n`` training samples.

    The tree has to store the number of training samples of its nodes
    (``TreeStructure.n_node_samples``, e.g., trees built from scikit-learn).
    Combine it with another target, e.g., ``OtherClass() & MinSamples(20)``.

    Attributes
    ----------
    n : int
        The smallest number of training samples (positive).
    """
    n: int

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(
                self.n, (int, np.integer)) or self.n < 1:
            raise ValueError('n has to be a positive integer; got '
                             '{!r}.'.format(self.n))

    def validate(self, context):
        if context.leaf_samples is None:
            raise ValueError('The tree does not store the number of training '
                             'samples of its leaves (n_node_samples).')

    def select(self, context):
        return context.leaf_samples >= self.n


def _targets(name, targets):
    if not targets:
        raise TypeError('{} needs at least one target.'.format(name))
    for target in targets:
        if not isinstance(target, Target):
            raise TypeError('{} combines targets; got {!r}.'.format(
                name, target))
    return tuple(targets)


@dataclasses.dataclass(frozen=True, init=False)
class AllOf(Target):
    """
    Leaves selected by all the targets (``a & b``).

    Attributes
    ----------
    targets : tuple of Target
    """
    targets: tuple

    def __init__(self, *targets):
        object.__setattr__(self, 'targets', _targets('AllOf', targets))

    def validate(self, context):
        for target in self.targets:
            target.validate(context)

    def select(self, context):
        mask = np.ones(context.leaf_ids.shape, dtype=bool)
        for target in self.targets:
            mask &= _mask(target, context)
        return mask


@dataclasses.dataclass(frozen=True, init=False)
class AnyOf(Target):
    """
    Leaves selected by any of the targets (``a | b``).

    Attributes
    ----------
    targets : tuple of Target
    """
    targets: tuple

    def __init__(self, *targets):
        object.__setattr__(self, 'targets', _targets('AnyOf', targets))

    def validate(self, context):
        for target in self.targets:
            target.validate(context)

    def select(self, context):
        mask = np.zeros(context.leaf_ids.shape, dtype=bool)
        for target in self.targets:
            mask |= _mask(target, context)
        return mask


@dataclasses.dataclass(frozen=True)
class Not(Target):
    """
    Leaves not selected by a target (``~a``).

    Attributes
    ----------
    target : Target
    """
    target: Target

    def __post_init__(self):
        _targets('Not', (self.target,))

    def validate(self, context):
        self.target.validate(context)

    def select(self, context):
        return ~_mask(self.target, context)


def _mask(target, context):
    """Selects the leaves of a (validated) target and checks the mask."""
    mask = np.asarray(target.select(context))
    if mask.dtype != bool or mask.shape != context.leaf_ids.shape:
        raise ValueError('{!r}.select has to return a boolean array with one '
                         'value per leaf ({}); got {} of shape {}.'.format(
                             target, context.leaf_ids.shape[0], mask.dtype,
                             mask.shape))
    return mask


def select_targets(target, context):
    """
    Validates a target and selects its leaves.

    Parameters
    ----------
    target : Target
    context : TargetContext

    Returns
    -------
    numpy.ndarray
        Boolean mask over ``context.leaf_ids``.
    """
    if not isinstance(target, Target):
        raise TypeError('target has to be a ctreex.targets.Target (e.g., '
                        'OtherClass(), TargetClass(...), SameClass() or '
                        'ValueChange(...)); got {!r}.'.format(target))
    target.validate(context)
    return _mask(target, context)
