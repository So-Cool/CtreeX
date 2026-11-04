"""
CtreeX: Counterfactual decision tree explainer.

The explainer turns the root-to-leaf paths of a decision tree into boolean
meta-features and finds the leaves of another class that are closest to a data
point.
It depends only on numpy.

>>> from ctreex import TreeCounterfactualExplainer
"""
from ctreex.explainer import (CounterfactualExplanation,
                              DistanceCounterfactuals,
                              TreeCounterfactualExplainer)
from ctreex.features import Feature, FeatureSpec
from ctreex.fixes import Fixes
from ctreex.tree import TreeStructure, load_tree

__version__ = '0.0.1'

__all__ = ['TreeCounterfactualExplainer', 'CounterfactualExplanation',
           'DistanceCounterfactuals', 'Feature', 'FeatureSpec', 'Fixes',
           'TreeStructure', 'load_tree']
