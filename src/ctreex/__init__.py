"""
CtreeX: Counterfactual decision tree explainer.

The explainer turns the root-to-leaf paths of a decision tree (a classifier
or a regressor) into boolean meta-features and finds the leaves -- e.g., of
another class -- that are closest to a data point.
It depends only on numpy.

>>> from ctreex import TreeCounterfactualExplainer
"""
from importlib.metadata import PackageNotFoundError, version

from ctreex.explainer import (CounterfactualExplanation,
                              DistanceCounterfactuals,
                              TreeCounterfactualExplainer)
from ctreex.features import Feature, FeatureSpec
from ctreex.audit import FeatureAudit, FeatureFlip, MonotonicityViolation
from ctreex.supportive import (ContextualisedCounterfactual,
                               SupportiveExplanation)
from ctreex.targets import (AllOf, AnyOf, ClassProbability, MinSamples, Not,
                            OtherClass, SameClass, Target, TargetClass,
                            ValueAtLeast, ValueAtMost, ValueChange,
                            ValueWithin)
from ctreex.tree import TreeStructure, load_tree

__all__ = ['TreeCounterfactualExplainer', 'CounterfactualExplanation',
           'DistanceCounterfactuals', 'Feature', 'FeatureSpec',
           'Target', 'OtherClass', 'TargetClass', 'SameClass',
           'ClassProbability', 'ValueChange', 'ValueAtLeast', 'ValueAtMost',
           'ValueWithin', 'MinSamples', 'AllOf', 'AnyOf', 'Not',
           'SupportiveExplanation', 'ContextualisedCounterfactual',
           'MonotonicityViolation', 'FeatureAudit', 'FeatureFlip',
           'TreeStructure', 'load_tree']

try:
    __version__ = version('ctreex')  # the version in pyproject.toml
except PackageNotFoundError:         # pragma: no cover (not installed)
    __version__ = 'unknown'
