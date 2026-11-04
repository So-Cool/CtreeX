"""
Helpers for scikit-learn decision trees (requires the ``sklearn`` extra).

Implements input validation, feature importance,
Graphviz/text/matplotlib representations of a tree and root-to-leaf paths.
"""
import re
import warnings

import numpy as np

__all__ = ['decision_tree_feature_importance', 'decision_tree_dot',
           'decision_tree_text', 'decision_tree_plot',
           'decision_tree_root2leaf', 'graphviz_tree']


def _sklearn():
    try:
        import sklearn.base
        import sklearn.tree
        import sklearn.utils.validation
    except ImportError as e:  # pragma: no cover
        raise ImportError('This functionality requires scikit-learn: '
                          'pip install "ctreex[sklearn]".') from e
    return sklearn


def _is_scikit_tree(clf):
    """Checks whether a model is a scikit-learn decision tree."""
    sklearn = _sklearn()
    if not isinstance(clf, sklearn.base.BaseEstimator):
        raise TypeError('Must be a valid scikit-learn model.')
    return isinstance(clf, (sklearn.tree.DecisionTreeClassifier,
                            sklearn.tree.DecisionTreeRegressor))


def _is_fitted_tree(clf):
    """Checks whether a scikit-learn decision tree is fitted (raises
    ``sklearn.exceptions.NotFittedError`` otherwise)."""
    sklearn = _sklearn()
    if not _is_scikit_tree(clf):
        raise TypeError('Has to be a scikit-learn decision tree.')
    sklearn.utils.validation.check_is_fitted(clf, 'tree_')
    return True


def _is_classifier_tree(clf):
    """``True`` for a tree classifier, ``False`` for a regressor."""
    sklearn = _sklearn()
    if not _is_scikit_tree(clf):
        raise TypeError('Has to be a scikit-learn decision tree.')
    return isinstance(clf, sklearn.tree.DecisionTreeClassifier)


def _validate_input(clf, feature_names, class_names, class_map):
    """Validates the parameters shared by the tree representations."""
    if not _is_scikit_tree(clf):
        raise TypeError('Has to be a scikit-learn decision tree.')
    _is_fitted_tree(clf)

    if feature_names is not None:
        if not isinstance(feature_names, list):
            raise TypeError('The feature names parameter has to be either a '
                            'list of strings or None.')
        if not feature_names:
            raise ValueError('The feature names list cannot be empty.')
        if not all(isinstance(i, str) for i in feature_names):
            raise TypeError('All feature names should be strings.')
        if len(feature_names) != clf.n_features_in_:
            raise ValueError('The feature names list has to name all {} '
                             'features.'.format(clf.n_features_in_))
    if class_names is not None:
        if not isinstance(class_names, list):
            raise TypeError('The class names parameter has to be either a '
                            'list of strings or None.')
        if not class_names:
            raise ValueError('The class names list cannot be empty.')
        if not all(isinstance(i, str) for i in class_names):
            raise TypeError('All class names should be strings.')
        if _is_classifier_tree(clf) and len(class_names) != len(clf.classes_):
            raise ValueError('The class names list has to name all {} '
                             'classes.'.format(len(clf.classes_)))
    if class_map is not None:
        if not isinstance(class_map, dict):
            raise TypeError('The class map parameter has to be either a '
                            'dictionary or None.')
        if not class_map:
            raise ValueError('The class map dictionary cannot be empty.')
        for i, v in class_map.items():
            if _is_classifier_tree(clf) and i not in clf.classes_:
                raise ValueError('The *{}* key in the class map dictionary is '
                                 'not a valid class id for the clf model.'
                                 ''.format(i))
            if not isinstance(v, str):
                raise TypeError('All class names in the class map should be '
                                'strings.')
        if (_is_classifier_tree(clf)
                and len(class_map) != len(clf.classes_)):
            raise ValueError('The class map has to name all the classes.')
    if class_names is not None and class_map is not None:
        warnings.warn('Both class names and a class map were given; the class '
                      'names take precedence.', UserWarning)
    return True


def _class_names(clf, class_names, class_map):
    if not _is_classifier_tree(clf):
        if class_names is not None or class_map is not None:
            warnings.warn('Class names or class map was given for a regressor '
                          'tree. These parameters will be ignored.',
                          UserWarning)
        return None
    if class_names is not None:
        return class_names
    if class_map is not None:
        return [class_map[i] for i in clf.classes_]
    return ['class {}'.format(i) for i in clf.classes_]


def decision_tree_feature_importance(clf):
    """Feature importances of a fitted scikit-learn decision tree."""
    _is_fitted_tree(clf)
    return clf.feature_importances_


def decision_tree_dot(clf, feature_names=None, class_names=None,
                      class_map=None, **kwargs):
    """
    Graphviz (DOT) representation of a scikit-learn tree.

    Wraps ``sklearn.tree.export_graphviz`` (``filled``, ``rounded`` and
    ``special_characters`` default to ``True``). Feature names default to
    ``feature <i>``; class names come from ``class_names``, ``class_map``
    (class label -> name) or default to ``class <label>``.
    """
    sklearn = _sklearn()
    _validate_input(clf, feature_names, class_names, class_map)
    if feature_names is None:
        feature_names = ['feature {}'.format(i)
                         for i in range(clf.n_features_in_)]
    kwargs.setdefault('filled', True)
    kwargs.setdefault('rounded', True)
    kwargs.setdefault('special_characters', True)
    kwargs['feature_names'] = feature_names
    kwargs['class_names'] = _class_names(clf, class_names, class_map)
    return sklearn.tree.export_graphviz(clf, **kwargs)


def decision_tree_text(clf, feature_names=None, class_names=None,
                       class_map=None, **kwargs):
    """Text representation of a tree (``sklearn.tree.export_text``) with
    class labels replaced by their names."""
    sklearn = _sklearn()
    _validate_input(clf, feature_names, class_names, class_map)
    if feature_names is None:
        feature_names = ['feature {}'.format(i)
                         for i in range(clf.n_features_in_)]
    tree_text = sklearn.tree.export_text(clf, feature_names=feature_names,
                                         **kwargs)
    names = _class_names(clf, class_names, class_map)
    if names is not None and (class_names is not None
                              or class_map is not None):
        mapping = {str(c): n for c, n in zip(clf.classes_, names)}
        regex_class = re.compile(r'(?P<beg>\sclass:\s)(?P<class>\S+)', re.I)
        tree_text = regex_class.sub(
            lambda m: '{}{}'.format(m.group('beg'),
                                    mapping.get(m.group('class'),
                                                m.group('class'))),
            tree_text)
    return tree_text


def decision_tree_plot(clf, feature_names=None, class_names=None,
                       class_map=None, **kwargs):
    """Plots a tree with ``sklearn.tree.plot_tree`` (returns the list of
    ``matplotlib.text.Annotation`` objects)."""
    sklearn = _sklearn()
    _validate_input(clf, feature_names, class_names, class_map)
    if feature_names is None:
        feature_names = ['feature {}'.format(i)
                         for i in range(clf.n_features_in_)]
    kwargs.setdefault('filled', True)
    kwargs.setdefault('rounded', True)
    return sklearn.tree.plot_tree(
        clf, feature_names=feature_names,
        class_names=_class_names(clf, class_names, class_map), **kwargs)


def decision_tree_root2leaf(tree, leaf_id):
    """
    Node ids on the path from the root to a leaf.

    Parameters
    ----------
    tree : scikit-learn tree or ctreex.tree.TreeStructure
    leaf_id : int

    Returns
    -------
    numpy.ndarray
        Node ids starting with the root (0) and ending with ``leaf_id``.
    """
    if hasattr(tree, 'tree_'):
        tree = tree.tree_
    left, right = np.asarray(tree.children_left), np.asarray(
        tree.children_right)
    if not 0 <= leaf_id < left.shape[0]:
        raise ValueError('Unknown node: {}.'.format(leaf_id))
    path = [leaf_id]
    node = leaf_id
    while node != 0:
        where_left = np.flatnonzero(left == node)
        where_right = np.flatnonzero(right == node)
        if where_left.size == 1 and where_right.size == 0:
            node = int(where_left[0])
        elif where_right.size == 1 and where_left.size == 0:
            node = int(where_right[0])
        else:
            raise ValueError('Malformed tree around node {}.'.format(node))
        path.append(node)
    return np.asarray(path[::-1])


def graphviz_tree(clf, feature_names, class_map=None):
    """
    Renders a tree with the ``graphviz`` package (original helper).

    Requires the ``graphviz`` Python package and the Graphviz binaries.
    """
    import html
    try:
        import graphviz
    except ImportError as e:  # pragma: no cover
        raise ImportError('graphviz_tree requires the graphviz package.') \
            from e
    class_names = ['{}'.format(i) for i in clf.classes_]
    if class_map is not None:
        class_names = [class_map[i] for i in class_names]
    feature_names = [html.escape(i, quote=True) for i in feature_names]
    dot_data = _sklearn().tree.export_graphviz(
        clf, out_file=None, feature_names=feature_names,
        class_names=class_names, filled=True, rounded=True,
        special_characters=True)
    return graphviz.Source(dot_data)
