"""Plain-text explanations."""
import numpy as np
import pytest

from ctreex import (SameClass, TreeCounterfactualExplainer, TreeStructure,
                    ValueChange, ValueWithin)
from ctreex import text


def test_helpers():
    assert text.join([]) == ''
    assert text.join(['a']) == 'a'
    assert text.join(['a', 'b']) == 'a and b'
    assert text.join(['a', 'b', 'c'], 'or') == 'a, b or c'
    assert text.format_value(2.50) == '2.5'
    assert text.format_value(3) == '3'
    assert text.format_value('x') == 'x'
    assert text.condition('age', {'is': [25], 'wannabe': ['more than 30']},
                          'been', with_value=True) == \
        'age been more than 30 (instead of 25)'
    assert text.supportive_sentence({}, 'yes') == \
        'The prediction is yes whatever the features.'


def test_explanation_text(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    x = toy_spec.encode({'age': 25, 'job': 'b'})
    explanation = explainer.explain_instance(x)
    assert explainer.explanation_text(explanation) == [
        'The prediction is no.',
        'Had job type been artist (instead of baker), the prediction would '
        'have been yes.',
        'Had age been between 30.0 and 50.0 (instead of 25), the prediction '
        'would have been yes.']
    assert explainer.explanation_text(explanation, n=1) == \
        explainer.explanation_text(explanation)[:2]
    assert explainer.explanation_text(explanation, integer_ranges=True)[
        2].startswith('Had age been between 30.0 and 50.0')
    same = explainer.explain_instance(x, target=SameClass())
    assert explainer.explanation_text(same)[1] == \
        'Had age been more than 50.0 (instead of 25), the prediction would ' \
        'still have been no.'


def test_contextualised_text():
    # A <= 5 -> c0 (leaf 1); A > 5: B <= 3 -> c0 (leaf 3), B > 3 -> c1
    tree = TreeStructure(
        [1, -1, 3, -1, -1], [2, -1, 4, -1, -1], [0, -2, 1, -2, -2],
        [5., -2, 3., -2, -2], [[1, 1], [1, 0], [1, 1], [1, 0], [0, 1]],
        ['c0', 'c1'], feature_names=['A', 'B'])
    explainer = TreeCounterfactualExplainer(tree)
    x = np.array([7., 1.])
    explanation = explainer.explain_instance(x)
    contextualised = explainer.contextualise(explanation)
    assert explainer.explanation_text(explanation, contextualised) == [
        'The prediction is c0.',
        'Had B been more than 3.0 (instead of 1), the prediction would have '
        'been c1, provided that A is more than 5.0.']
    assert explainer.supportive_text(np.array([2., 1.])) == [
        'The prediction is c0 since A is less or equal to 5.0, whatever the '
        'other features.',
        'The prediction is c0 since B is less or equal to 3.0, whatever the '
        'other features.']


def test_regression_text():
    tree = TreeStructure(
        children_left=[1, -1, 3, -1, -1], children_right=[2, -1, 4, -1, -1],
        feature=[0, -2, 0, -2, -2], threshold=[10., -2, 20., -2, -2],
        value=[[150.], [100.], [200.], [150.], [300.]], feature_names=['x'])
    explainer = TreeCounterfactualExplainer(tree)
    x = np.array([5.])
    explanation = explainer.explain_instance(x, target=ValueChange(+100))
    assert explainer.explanation_text(explanation) == [
        'The prediction is 100.',
        'Had x been more than 20.0 (instead of 5), the prediction would have '
        'been 300 (instead of 100).']
    assert explainer.supportive_text(x, target=ValueWithin(10)) == [
        'The prediction is within 10 of 100 since x is less or equal to 10.0,'
        ' whatever the other features.']
    assert explainer.supportive_text(x, target=ValueWithin(500)) == [
        'The prediction is within 500 of 100 whatever the features.']


def test_hidden_thresholds_text(toy_tree, toy_spec):
    explainer = TreeCounterfactualExplainer(toy_tree, toy_spec)
    toy_spec.features['age'].minimum, toy_spec.features['age'].maximum = \
        18, 90
    x = toy_spec.encode({'age': 25, 'job': 'b'})
    explanation = explainer.explain_instance(x)
    hidden = explainer.explanation_text(explanation, hide_thresholds=True,
                                        random_state=0)
    assert hidden == explainer.explanation_text(
        explanation, hide_thresholds=True, random_state=0)
    assert 'between' not in hidden[2] and '30.0' not in hidden[2]
    age = float(hidden[2].split('been ')[1].split(' ')[0])
    assert 30 < age <= 50
    with pytest.raises(ValueError):  # no maximum
        toy_spec.features['age'].maximum = None
        explainer.explanation_text(
            explainer.explain_instance(x, target=SameClass()),
            hide_thresholds=True)
