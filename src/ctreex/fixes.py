"""
Switches between the legacy (2018) behaviour and fixed behaviour.

Every behaviour-changing fix is controlled by a separate boolean field of
:class:`Fixes`. ``Fixes.legacy()`` reproduces the original (2018) implementation
of the algorithm; ``Fixes.fixed()`` enables all the fixes.
A single fix can be toggled with :func:`dataclasses.replace`, e.g.,::

    fixes = dataclasses.replace(Fixes.legacy(), deterministic_order=True)

Crash fixes are always applied and are therefore not listed here;
refer to the changelog in ``README.md`` for more details.
"""
import dataclasses

__all__ = ['Fixes', 'resolve_fixes', 'IMPLEMENTATIONS']

IMPLEMENTATIONS = ('legacy', 'fixed')


@dataclasses.dataclass(frozen=True)
class Fixes(object):
    """
    Behaviour-changing fixes of the counterfactual explainer.

    Attributes
    ----------
    deterministic_order : bool
        The original implementation de-duplicates the alternative values of
        categorical features (and the current value of a feature) with
        ``list(set(...))``. The order of strings in a set depends on Python's
        (randomised) string hashing, so the order of the alternatives changed
        between program runs. When ``True``, duplicates are removed while
        keeping the order in which the values were found.
    continue_past_missing_distance : bool
        The original implementation collects counterfactuals of distance
        1, 2, ... and stops at the first distance at which *no* leaf exists
        (a ``KeyError``), even when leaves at larger distances exist. When
        ``True`` missing distances are skipped and all distances up to the
        maximum are examined.
    complete_numerical_intervals : bool
        The original implementation describes a numerical feature split at a
        single threshold ``t`` only with the ``A<=t`` interval, while features
        with ``k`` thresholds get ``k + 1`` intervals. When ``True``, the
        missing ``t<A`` interval is added. The counterfactuals of data points
        and their distances to the leaves do not change (only the internal
        representation of the "more than t" intervals); the leaf-to-leaf
        distances -- hence the volatility index and the leaf support sets --
        do, since every interval contributes to them.
    whole_counterfactual_restrictions : bool
        How the ``excluded``/``included`` features of
        ``explain_instance`` are applied. The original implementation removes
        the forbidden features from the counterfactual of every leaf, which
        yields incomplete counterfactuals (changing only the remaining
        features does not reach the leaf, e.g., a distance-2 counterfactual
        listing one feature) and empty ones. When ``True``, leaves whose
        counterfactual needs a forbidden feature are skipped, so every
        counterfactual at distance ``d`` changes exactly ``d`` features.
    """
    deterministic_order: bool = True
    continue_past_missing_distance: bool = True
    complete_numerical_intervals: bool = True
    whole_counterfactual_restrictions: bool = True

    @classmethod
    def legacy(cls):
        """All fixes disabled; the behaviour of the 2018 implementation."""
        return cls(**{f.name: False for f in dataclasses.fields(cls)})

    @classmethod
    def fixed(cls):
        """All fixes enabled."""
        return cls(**{f.name: True for f in dataclasses.fields(cls)})


def resolve_fixes(implementation='fixed', fixes=None):
    """
    Combines the ``implementation`` switch with optional per-fix overrides.

    Parameters
    ----------
    implementation : {'legacy', 'fixed'}
        The base behaviour.
    fixes : Fixes or dict, optional
        Either a complete :class:`Fixes` object (``implementation`` is then
        ignored) or a dictionary overriding selected fields.

    Returns
    -------
    Fixes
        The resolved switches.
    """
    if implementation not in IMPLEMENTATIONS:
        raise ValueError('implementation has to be one of {}; got {!r}.'
                         ''.format(IMPLEMENTATIONS, implementation))
    if isinstance(fixes, Fixes):
        return fixes
    base = Fixes.legacy() if implementation == 'legacy' else Fixes.fixed()
    if fixes is None:
        return base
    if not isinstance(fixes, dict):
        raise TypeError('fixes has to be a Fixes object, a dictionary or '
                        'None.')
    return dataclasses.replace(base, **fixes)
