"""
Plain-text explanations from a generic template.

The sentences are built from the human-readable descriptions of
:meth:`ctreex.explainer.TreeCounterfactualExplainer.describe` -- the display
names of the features, the names of their categories and the names of the
classes -- so they need no templates per data set, e.g.,
"Had duration been less or equal to 22.5 (instead of 48), the prediction
would have been low risk."

* :func:`counterfactual_sentence` -- a counterfactual (with an optional
  context);
* :func:`supportive_sentence` -- a supportive explanation; and
* :func:`prediction_sentence` -- the prediction.
"""

__all__ = ['format_value', 'join', 'condition', 'prediction_sentence',
           'counterfactual_sentence', 'supportive_sentence']


def format_value(value):
    """Numbers in their shortest form (``{:g}``), other values as text."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return '{:g}'.format(value)
    return str(value)


def join(parts, word='and'):
    """``'a'``, ``'a and b'``, ``'a, b and c'``."""
    parts = list(parts)
    if len(parts) < 2:
        return ''.join(parts)
    return '{} {} {}'.format(', '.join(parts[:-1]), word, parts[-1])


def _values(values):
    return join([format_value(v) for v in values], 'or')


def condition(name, described, verb='is', with_value=False):
    """
    ``'<feature> <verb> <values>'``, e.g., ``'duration is at most 22.5'``.

    Parameters
    ----------
    name : str
        The display name of the feature.
    described : dict
        ``{'is': [...], 'wannabe': [...]}`` (from ``describe``).
    verb : str
    with_value : bool
        Add the current value, e.g., ``' (instead of 48)'``.
    """
    text = '{} {} {}'.format(name, verb, _values(described['wannabe']))
    if with_value and described['is']:
        text += ' (instead of {})'.format(_values(described['is']))
    return text


def prediction_sentence(prediction):
    """``'The prediction is <prediction>.'``"""
    return 'The prediction is {}.'.format(format_value(prediction))


def counterfactual_sentence(change, outcome, context=None):
    """
    ``'Had <changes>, the prediction would have been <outcome>[, provided
    that <context>].'``

    Parameters
    ----------
    change : dict
        ``{feature name: {'is': [...], 'wannabe': [...]}}`` (from
        ``describe``).
    outcome : str
        E.g., ``'the prediction would have been low risk'``.
    context : dict, optional
        The context of a contextualised counterfactual (same format).
    """
    sentence = 'Had {}, {}'.format(
        join(condition(name, part, 'been', with_value=True)
             for name, part in change.items()), outcome)
    if context:
        sentence += ', provided that {}'.format(join(
            condition(name, part) for name, part in context.items()))
    return sentence + '.'


def supportive_sentence(conditions, prediction):
    """
    ``'The prediction is <prediction> since <conditions>, whatever the other
    features.'``

    Parameters
    ----------
    conditions : dict
        ``{feature name: {'is': [...], 'wannabe': [...]}}`` (from
        ``describe``); empty when the prediction does not depend on the
        features.
    prediction : str
        The (formatted) guaranteed outcome, e.g., ``'low risk'``.
    """
    if not conditions:
        return 'The prediction is {} whatever the features.'.format(
            prediction)
    return 'The prediction is {} since {}, whatever the other ' \
        'features.'.format(prediction, join(
            condition(name, part) for name, part in conditions.items()))
