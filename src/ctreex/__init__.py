"""
CtreeX: Counterfactual decision tree explainer.

The explainer turns the root-to-leaf paths of a decision tree into boolean
meta-features and finds the leaves of another class that are closest to a data
point.
It depends only on numpy.

>>> from ctreex import TreeCounterfactualExplainer
"""
from importlib.metadata import PackageNotFoundError, version

from ctreex.explainer import (CounterfactualExplanation,
                              DistanceCounterfactuals,
                              TreeCounterfactualExplainer)
from ctreex.features import Feature, FeatureSpec
from ctreex.tree import TreeStructure, load_tree

__all__ = ['TreeCounterfactualExplainer', 'CounterfactualExplanation',
           'DistanceCounterfactuals', 'Feature', 'FeatureSpec',
           'TreeStructure', 'load_tree']

try:
    __version__ = version('ctreex')  # the version in pyproject.toml
except PackageNotFoundError:         # pragma: no cover (not installed)
    __version__ = 'unknown'
