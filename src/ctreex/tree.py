"""
A library-independent representation of a fitted decision tree classifier.

:class:`TreeStructure` stores the arrays of a binary decision tree (the same
layout as ``sklearn.tree.DecisionTreeClassifier.tree_``) and can be built
from a scikit-learn model, saved to and loaded from JSON. Storing trees as JSON
makes them independent of the scikit-learn version (pickled scikit-learn
models can only be loaded by the version that created them).
"""
import json
import os

import numpy as np

from ctreex import _validation as val

__all__ = ['TreeStructure', 'load_tree', 'TREE_LEAF', 'TREE_UNDEFINED']

TREE_LEAF = -1          #: child id of leaves (as in scikit-learn)
TREE_UNDEFINED = -2     #: feature id of leaves (as in scikit-learn)
JSON_FORMAT = 'ctreex-tree'
JSON_FORMAT_VERSION = 1


class TreeStructure(object):
    """
    A fitted binary decision tree classifier.

    A data point goes to the left child of a split node when
    ``x[feature] <= threshold`` and to the right child otherwise.

    Parameters
    ----------
    children_left, children_right : array-like of int
        Child node ids (``-1`` for leaves).
    feature : array-like of int
        Column used by a split node (``-2`` for leaves).
    threshold : array-like of float
        Split thresholds.
    value : array-like of float, shape (n_nodes, n_classes)
        (Weighted) class counts of the training data in every node.
    classes : array-like
        Class labels (in the order of ``value`` columns).
    n_features : int, optional
        Number of columns of the data (inferred when not given).
    feature_names : list of str, optional
        Names of the data columns.
    metadata : dict, optional
        Any additional (JSON-serialisable) information.
    """

    def __init__(self, children_left, children_right, feature, threshold,
                 value, classes, n_features=None, feature_names=None,
                 metadata=None):
        self.children_left = np.asarray(children_left, dtype=np.int64)
        self.children_right = np.asarray(children_right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 3:  # scikit-learn layout (n_nodes, n_outputs, ...)
            if value.shape[1] != 1:
                raise ValueError('Multi-output trees are not supported.')
            value = value[:, 0, :]
        self.value = value
        self.classes = np.asarray(classes)
        n_nodes = self.children_left.shape[0]
        for name in ('children_right', 'feature', 'threshold', 'value'):
            if getattr(self, name).shape[0] != n_nodes:
                raise ValueError('All the tree arrays have to describe {} '
                                 'nodes ({} differs).'.format(n_nodes, name))
        if self.value.shape[1] != self.classes.shape[0]:
            raise ValueError('The number of classes does not match the shape '
                             'of the value array.')
        is_leaf = self.children_left == TREE_LEAF
        if not np.array_equal(is_leaf, self.children_right == TREE_LEAF):
            raise ValueError('Every node needs either two or zero children.')
        if n_features is None:
            n_features = int(self.feature[~is_leaf].max()) + 1 \
                if (~is_leaf).any() else 0
        self.n_features = int(n_features)
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError('feature_names has to name {} columns.'.format(
                n_features))
        self.feature_names = (None if feature_names is None
                              else [str(i) for i in feature_names])
        self.metadata = dict(metadata or {})

    # ------------------------------------------------------------------ build
    @classmethod
    def from_sklearn(cls, clf, feature_names=None, metadata=None):
        """
        Builds the structure from a fitted scikit-learn decision tree.

        Objects exposing the tree as a ``model`` attribute (e.g. an
        ``mlaas.model_trainer.Trainer``) are accepted as well.
        """
        if hasattr(clf, 'model') and not hasattr(clf, 'tree_'):
            clf = clf.model
        if not hasattr(clf, 'tree_') or not hasattr(clf, 'classes_'):
            raise TypeError('Expected a fitted scikit-learn decision tree '
                            'classifier.')
        tree = clf.tree_
        if feature_names is None and hasattr(clf, 'feature_names_in_'):
            feature_names = list(clf.feature_names_in_)
        meta = {'source': 'scikit-learn',
                'params': {k: v for k, v in clf.get_params().items()
                           if isinstance(v, (int, float, str, type(None)))}}
        meta.update(metadata or {})
        return cls(tree.children_left, tree.children_right, tree.feature,
                   tree.threshold, tree.value, clf.classes_,
                   n_features=tree.n_features, feature_names=feature_names,
                   metadata=meta)

    @classmethod
    def from_dict(cls, d):
        """Builds the structure from :meth:`to_dict` output."""
        if d.get('format') != JSON_FORMAT:
            raise ValueError('Not a ctreex tree description.')
        if d.get('format_version', 0) > JSON_FORMAT_VERSION:
            raise ValueError('Unsupported tree format version.')
        metadata = {k: v for k, v in d.items() if k not in (
            'format', 'format_version', 'children_left', 'children_right',
            'feature', 'threshold', 'value', 'classes', 'n_features',
            'feature_names')}
        return cls(d['children_left'], d['children_right'], d['feature'],
                   d['threshold'], d['value'], d['classes'],
                   n_features=d.get('n_features'),
                   feature_names=d.get('feature_names'), metadata=metadata)

    def to_dict(self):
        d = {'format': JSON_FORMAT, 'format_version': JSON_FORMAT_VERSION,
             'n_features': self.n_features,
             'classes': self.classes.tolist(),
             'feature_names': self.feature_names,
             'children_left': self.children_left.tolist(),
             'children_right': self.children_right.tolist(),
             'feature': self.feature.tolist(),
             'threshold': self.threshold.tolist(),
             'value': self.value.tolist()}
        d.update(self.metadata)
        return d

    def to_json(self, path=None, indent=None):
        """Serialises the tree to a JSON string (and a file if given)."""
        payload = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, 'w') as f:
                f.write(payload)
        return payload

    # -------------------------------------------------------------- inspect
    @property
    def n_nodes(self):
        return self.children_left.shape[0]

    @property
    def is_leaf(self):
        return self.children_left == TREE_LEAF

    @property
    def leaves(self):
        """Ids of the leaves (in ascending order)."""
        return np.flatnonzero(self.is_leaf)

    def leaf_class(self, leaf):
        """Majority class of a node (ties go to the first class)."""
        return self.classes[int(np.argmax(self.value[leaf]))]

    def apply(self, X):
        """Returns the id of the leaf each data point falls into."""
        X = val.check_array_2d(X, self.n_features)
        leaves = np.empty(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            node = 0
            while self.children_left[node] != TREE_LEAF:
                if x[self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            leaves[i] = node
        return leaves

    def predict(self, X):
        """Predicts the (majority) class of each data point."""
        return np.asarray([self.leaf_class(l) for l in self.apply(X)])

    def predict_proba(self, X):
        value = self.value[self.apply(X)]
        return value / value.sum(axis=1, keepdims=True)

    def __repr__(self):
        return 'TreeStructure(n_nodes={}, n_leaves={}, n_features={}, ' \
               'classes={})'.format(self.n_nodes, self.leaves.shape[0],
                                    self.n_features, self.classes.tolist())


def load_tree(path_or_object, feature_names=None):
    """
    Loads a tree from a JSON file, a (modern) pickle or a fitted model.

    Parameters
    ----------
    path_or_object : str, os.PathLike, dict or object
        A ``.json`` file written by :meth:`TreeStructure.to_json`, a pickle
        (``.pkl``/``.joblib``) of a scikit-learn decision tree (or of an object
        holding it in a ``model`` attribute), a dictionary or a fitted model.

    Returns
    -------
    TreeStructure
    """
    if isinstance(path_or_object, TreeStructure):
        return path_or_object
    if isinstance(path_or_object, dict):
        return TreeStructure.from_dict(path_or_object)
    if isinstance(path_or_object, (str, os.PathLike)):
        path = os.fspath(path_or_object)
        if path.endswith('.json'):
            with open(path) as f:
                return TreeStructure.from_dict(json.load(f))
        import joblib  # scikit-learn depends on joblib
        model = joblib.load(path)
        return TreeStructure.from_sklearn(model, feature_names=feature_names)
    return TreeStructure.from_sklearn(path_or_object,
                                      feature_names=feature_names)
