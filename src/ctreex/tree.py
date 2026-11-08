"""
A library-independent representation of a fitted decision tree.

:class:`TreeStructure` stores the arrays of a binary decision tree (the same
layout as ``sklearn.tree.DecisionTreeClassifier.tree_`` and
``sklearn.tree.DecisionTreeRegressor.tree_``) and can be built from a
scikit-learn model, saved to and loaded from JSON.
Storing trees as JSON makes them independent of the scikit-learn version
(pickled scikit-learn models can only be loaded by the version that created
them).
"""
import json
import os

import numpy as np

from ctreex import _validation as val

__all__ = ['TreeStructure', 'load_tree', 'TREE_LEAF', 'TREE_UNDEFINED']

TREE_LEAF = -1          #: child id of leaves (as in scikit-learn)
TREE_UNDEFINED = -2     #: feature id of leaves (as in scikit-learn)
JSON_FORMAT = 'ctreex-tree'
JSON_FORMAT_VERSION = 2
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
TASKS = (CLASSIFICATION, REGRESSION)


class TreeStructure(object):
    """
    A fitted binary decision tree -- a classifier or a regressor.

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
    value : array-like of float, shape (n_nodes, n_classes) or (n_nodes, 1)
        Classifiers: (weighted) class counts -- or, for trees built with
        scikit-learn 1.4 or newer, class proportions -- of the training data
        in every node;
        only their relative sizes are used (predictions and probabilities).
        Regressors: the prediction of every node (a single column).
    classes : array-like, optional
        Class labels (in the order of ``value`` columns);
        ``None`` for regression trees.
    n_features : int, optional
        Number of columns of the data (inferred when not given).
    feature_names : list of str, optional
        Names of the data columns.
    metadata : dict, optional
        Any additional (JSON-serialisable) information.
    missing_go_to_left : array-like of bool, optional
        Where missing values (``nan``) go at every split node -- learnt by
        scikit-learn 1.3 or newer. When not given, missing values go to the
        right child (as with scikit-learn trees trained without them).

    Multi-output trees are not supported.
    """

    def __init__(self, children_left, children_right, feature, threshold,
                 value, classes=None, n_features=None, feature_names=None,
                 metadata=None, missing_go_to_left=None):
        self.children_left = np.asarray(children_left, dtype=np.int64)
        self.children_right = np.asarray(children_right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 3:  # scikit-learn layout (n_nodes, n_outputs, ...)
            if value.shape[1] != 1:
                raise ValueError('Multi-output trees are not supported.')
            value = value[:, 0, :]
        if value.ndim != 2:
            raise ValueError('value has to be a (n_nodes, n_classes) or '
                             '(n_nodes, 1) array.')
        self.value = value
        self.classes = None if classes is None else np.asarray(classes)
        self.missing_go_to_left = (
            None if missing_go_to_left is None
            else np.asarray(missing_go_to_left, dtype=bool))
        n_nodes = self.children_left.shape[0]
        if (self.missing_go_to_left is not None
                and self.missing_go_to_left.shape[0] != n_nodes):
            raise ValueError('missing_go_to_left has to describe {} '
                             'nodes.'.format(n_nodes))
        for name in ('children_right', 'feature', 'threshold', 'value'):
            if getattr(self, name).shape[0] != n_nodes:
                raise ValueError('All the tree arrays have to describe {} '
                                 'nodes ({} differs).'.format(n_nodes, name))
        if self.classes is None:
            if self.value.shape[1] != 1:
                raise ValueError('Regression trees (without classes) have a '
                                 'single value per node; for classifiers, '
                                 'give the classes.')
        elif self.value.shape[1] != self.classes.shape[0]:
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

        Classifiers and regressors (e.g., ``DecisionTreeClassifier`` and
        ``DecisionTreeRegressor``) are supported.
        Objects exposing the tree as a ``model`` attribute (e.g. an
        ``mlaas.model_trainer.Trainer``) are accepted as well.
        """
        if hasattr(clf, 'model') and not hasattr(clf, 'tree_'):
            clf = clf.model
        if not hasattr(clf, 'tree_'):
            raise TypeError('Expected a fitted scikit-learn decision tree '
                            '(classifier or regressor).')
        tree = clf.tree_
        if getattr(tree, 'n_outputs', 1) != 1:
            raise ValueError('Multi-output trees are not supported.')
        if feature_names is None and hasattr(clf, 'feature_names_in_'):
            feature_names = list(clf.feature_names_in_)
        meta = {'source': 'scikit-learn',
                'params': {k: v for k, v in clf.get_params().items()
                           if isinstance(v, (int, float, str, type(None)))}}
        meta.update(metadata or {})
        return cls(tree.children_left, tree.children_right, tree.feature,
                   tree.threshold, tree.value, getattr(clf, 'classes_', None),
                   n_features=tree.n_features, feature_names=feature_names,
                   metadata=meta,
                   missing_go_to_left=getattr(tree, 'missing_go_to_left',
                                              None))

    @classmethod
    def from_dict(cls, d):
        """Builds the structure from :meth:`to_dict` output."""
        if d.get('format') != JSON_FORMAT:
            raise ValueError('Not a ctreex tree description.')
        if d.get('format_version', 0) > JSON_FORMAT_VERSION:
            raise ValueError('Unsupported tree format version.')
        task = d.get('task', CLASSIFICATION)  # version 1: classifiers only
        if task not in TASKS:
            raise ValueError('Unknown task: {!r}.'.format(task))
        metadata = {k: v for k, v in d.items() if k not in (
            'format', 'format_version', 'task', 'children_left',
            'children_right', 'feature', 'threshold', 'value', 'classes',
            'n_features', 'feature_names', 'missing_go_to_left')}
        classes = d['classes'] if task == CLASSIFICATION else None
        if task == CLASSIFICATION and classes is None:
            raise ValueError('Classification trees need classes.')
        return cls(d['children_left'], d['children_right'], d['feature'],
                   d['threshold'], d['value'], classes,
                   n_features=d.get('n_features'),
                   feature_names=d.get('feature_names'), metadata=metadata,
                   missing_go_to_left=d.get('missing_go_to_left'))

    def to_dict(self):
        d = {'format': JSON_FORMAT, 'format_version': JSON_FORMAT_VERSION,
             'task': self.task,
             'n_features': self.n_features,
             'classes': (None if self.classes is None
                         else self.classes.tolist()),
             'feature_names': self.feature_names,
             'children_left': self.children_left.tolist(),
             'children_right': self.children_right.tolist(),
             'feature': self.feature.tolist(),
             'threshold': self.threshold.tolist(),
             'value': self.value.tolist()}
        if self.missing_go_to_left is not None:
            d['missing_go_to_left'] = self.missing_go_to_left.tolist()
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
    def task(self):
        """``'classification'`` or ``'regression'`` (without classes)."""
        return REGRESSION if self.classes is None else CLASSIFICATION

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
        if self.classes is None:
            raise TypeError('Regression trees have no classes; use '
                            'leaf_value.')
        return self.classes[int(np.argmax(self.value[leaf]))]

    def leaf_value(self, leaf):
        """Prediction of a node of a regression tree."""
        if self.classes is not None:
            raise TypeError('Classification trees predict classes; use '
                            'leaf_class.')
        return float(self.value[leaf, 0])

    def apply(self, X):
        """
        Returns the id of the leaf each data point falls into.

        Missing values (``nan``) follow ``missing_go_to_left`` (the right
        child when it is not available), as in scikit-learn.
        """
        X = val.check_array_2d(X, self.n_features).astype(np.float64)
        leaves = np.empty(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            node = 0
            while self.children_left[node] != TREE_LEAF:
                value = x[self.feature[node]]
                if np.isnan(value):
                    go_left = (self.missing_go_to_left is not None
                               and self.missing_go_to_left[node])
                    node = (self.children_left[node] if go_left
                            else self.children_right[node])
                elif value <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            leaves[i] = node
        return leaves

    def predict(self, X):
        """Predicts the (majority) class -- or, for regression trees, the
        value -- of each data point."""
        leaves = self.apply(X)
        if self.classes is None:
            return self.value[leaves, 0]
        return np.asarray([self.leaf_class(leaf) for leaf in leaves])

    def predict_proba(self, X):
        """Class probabilities of each data point (classifiers only)."""
        if self.classes is None:
            raise TypeError('Regression trees do not predict probabilities.')
        value = self.value[self.apply(X)]
        return value / value.sum(axis=1, keepdims=True)

    def __repr__(self):
        if self.classes is None:
            outputs = 'task=regression'
        else:
            outputs = 'classes={}'.format(self.classes.tolist())
        return 'TreeStructure(n_nodes={}, n_leaves={}, n_features={}, ' \
               '{})'.format(self.n_nodes, self.leaves.shape[0],
                            self.n_features, outputs)


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
