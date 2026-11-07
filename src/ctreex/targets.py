"""
Targets: which leaves the explanations of a data point lead to.

The explainer finds the leaves closest to a data point among the *target*
leaves and turns their conditions into explanations.
A target selects these leaves:

* :class:`OtherClass` (the default) -- leaves of any class other than the
  prediction, i.e., *class-contrastive* counterfactuals;
* :class:`TargetClass` -- leaves of one given class (multi-class trees); and
* :class:`SameClass` -- other leaves of the predicted class, i.e.,
  *class-equivalent* explanations:
  the changes that keep the prediction.

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

import numpy as np

__all__ = ['TargetContext', 'Target', 'OtherClass', 'TargetClass',
           'SameClass', 'select_targets']


@dataclasses.dataclass(frozen=True)
class TargetContext(object):
    """
    The prediction of a data point and the leaves of the tree.

    Attributes
    ----------
    prediction : object
        The predicted class (its name when the explainer has class names).
    leaf : int
        The leaf the data point falls into.
    leaf_ids : numpy.ndarray
        All the leaf ids
        (in the order of ``TreeCounterfactualExplainer.leaf_meta_features``).
    leaf_classes : numpy.ndarray
        The (majority) class of every leaf.
    leaf_probabilities : numpy.ndarray
        ``(n_leaves, n_classes)`` class proportions of every leaf.
    classes : list
        The classes of the columns of ``leaf_probabilities``.
    """
    prediction: object
    leaf: int
    leaf_ids: np.ndarray
    leaf_classes: np.ndarray
    leaf_probabilities: np.ndarray
    classes: list


class Target(object):
    """
    Base class of targets.

    Subclasses implement :meth:`select` and may implement :meth:`validate`.
    """

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

    def select(self, context):
        return context.leaf_classes != context.prediction


@dataclasses.dataclass(frozen=True)
class TargetClass(Target):
    """
    Leaves of a given class, which differs from the prediction.

    Attributes
    ----------
    label : object
        The class (its name when the explainer has class names).
    """
    label: object

    def validate(self, context):
        if self.label not in set(context.leaf_classes.tolist()):
            raise ValueError('Unknown target class (no leaf predicts it): '
                             '{!r}.'.format(self.label))
        if self.label == context.prediction:
            raise ValueError('The target class has to differ from the '
                             'prediction ({!r}); use SameClass() for '
                             'class-equivalent explanations.'.format(
                                 context.prediction))

    def select(self, context):
        return context.leaf_classes == self.label


@dataclasses.dataclass(frozen=True)
class SameClass(Target):
    """Other leaves of the predicted class (class-equivalent)."""

    def select(self, context):
        return context.leaf_classes == context.prediction


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
                        'OtherClass(), TargetClass(...) or SameClass()); '
                        'got {!r}.'.format(target))
    target.validate(context)
    mask = np.asarray(target.select(context))
    if mask.dtype != bool or mask.shape != context.leaf_ids.shape:
        raise ValueError('{!r}.select has to return a boolean array with one '
                         'value per leaf ({}); got {} of shape {}.'.format(
                             target, context.leaf_ids.shape[0], mask.dtype,
                             mask.shape))
    return mask
