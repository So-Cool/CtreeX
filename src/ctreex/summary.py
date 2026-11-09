"""
Global summaries: a few general rules describing every outcome of a tree.

The leaves of a tree partition the feature space into many small regions;
a summary describes the region of every outcome (class, or band of
predictions for regression trees) with a small number of larger, possibly
overlapping, rules, e.g.,
"IF checking account is no checking account THEN the prediction is low risk".

The rules are exact -- every data point satisfying a rule gets its outcome --
and together they cover all the leaves of the outcome:

1. every leaf of the outcome is grown into a rule, one feature at a time (in
   the order of ``LeafMetaFeatures.groups``), with a numerical feature
   extending its interval to the neighbouring intervals and a categorical
   feature adding categories, as long as every leaf the rule overlaps has
   the outcome; and
2. a greedy set cover picks the rule containing the most leaves not yet covered
   (in case of ties, more training samples and fewer conditions are
   prioritised) until every leaf of the outcome is contained in a rule.

The set cover is not guaranteed to be the smallest possible (finding it is
computationally hard).
"""
import dataclasses
import heapq

import numpy as np

from ctreex import counterfactuals as cfs
from ctreex.distances import allowed_rows
from ctreex.features import NUMERICAL

__all__ = ['Rule', 'OutcomeSummary', 'grow_rule', 'summarise']


@dataclasses.dataclass
class Rule(object):
    """
    A rule describing (part of) the region of an outcome.

    Attributes
    ----------
    outcome : object
        The class -- or, for regression trees, the band of predictions --
        of the data points satisfying the rule.
    conditions : dict
        ``{feature: {'is': [], 'wannabe': [...]}}`` -- the values of the
        features the rule restricts (the counterfactual format, so
        ``TreeCounterfactualExplainer.describe({0: conditions})`` works);
        empty when the whole feature space has the outcome.
    leaves : list of int
        The leaves contained in the rule (all of them have the outcome).
    samples : int or None
        The number of training samples in these leaves (``None`` when the
        tree does not store it).
    purity : float or None
        Classification trees: the proportion of the training samples in
        these leaves that are of the class (``None`` without the numbers of
        samples).
    values : tuple or None
        Regression trees: the smallest and largest predictions of these
        leaves.
    """
    outcome: object
    conditions: dict
    leaves: list
    samples: int = None
    purity: float = None
    values: tuple = None


@dataclasses.dataclass
class OutcomeSummary(object):
    """
    The rules describing one outcome.

    Attributes
    ----------
    outcome : object
    rules : list of Rule
        In the order they were picked (most leaves first).
    leaves : list of int
        All the leaves with the outcome.
    uncovered : list of int
        The leaves of the outcome not contained in any rule (only when the
        number of rules is limited).
    """
    outcome: object
    rules: list
    leaves: list
    uncovered: list


def _overlap(allows, box):
    """``(n_groups, n_leaves)``: whether every leaf overlaps the box on every
    feature."""
    return np.array([(a & b).any(axis=1) for a, b in zip(allows, box)])


def grow_rule(lmf, allows, positive, row):
    """
    Grows the region of a leaf into a rule.

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    allows : list of numpy.ndarray
        :func:`ctreex.distances.allowed_rows` of every group.
    positive : numpy.ndarray
        Boolean mask of the leaves with the outcome.
    row : int
        The row of the leaf (following ``lmf.leaf_ids``);
        it has to have the outcome.

    Returns
    -------
    list of numpy.ndarray
        The allowed meta-features of every group (the rule).
    """
    box = [a[row].copy() for a in allows]
    overlap = _overlap(allows, box)
    for g, (kind, _, _) in enumerate(lmf.groups):
        others = np.delete(overlap, g, axis=0).all(axis=0) \
            if len(box) > 1 else np.ones(positive.shape, dtype=bool)
        # meta-features of this feature that would add a leaf without the
        # outcome
        bad = allows[g][others & ~positive].any(axis=0)
        mask = box[g]
        if kind == NUMERICAL:
            selected = np.flatnonzero(mask)
            low, high = selected[0], selected[-1]
            while low > 0 and not bad[low - 1]:
                low -= 1
            while high < mask.size - 1 and not bad[high + 1]:
                high += 1
            mask[low:high + 1] = True
        else:
            mask |= ~bad
        overlap[g] = (allows[g] & mask).any(axis=1)
    return box


def _contained(allows, box):
    """Whether every leaf lies inside the box."""
    return ~np.any([(a & ~b).any(axis=1) for a, b in zip(allows, box)],
                   axis=0)


def _conditions(lmf, box):
    conditions = {}
    for (kind, feature, indices), mask in zip(lmf.groups, box):
        if mask.all():
            continue
        members = [lmf.meta_features[i] for i in indices]
        if kind == NUMERICAL:
            wannabe = cfs.mask_intervals(members, mask)
        else:
            wannabe = [m.category for m, keep in zip(members, mask) if keep]
        conditions[feature] = {'is': [], 'wannabe': wannabe}
    return conditions


def summarise(lmf, outcome, positive, samples=None, probabilities=None,
              values=None, max_rules=None):
    """
    The rules describing one outcome (see the module description).

    Parameters
    ----------
    lmf : ctreex.meta_features.LeafMetaFeatures
    outcome : object
        The class or band the rules report.
    positive : numpy.ndarray
        Boolean mask of the leaves with the outcome.
    samples : numpy.ndarray, optional
        The number of training samples of every leaf.
    probabilities : numpy.ndarray, optional
        Classification trees: the proportion of the class in every leaf.
    values : numpy.ndarray, optional
        Regression trees: the prediction of every leaf.
    max_rules : int, optional
        The largest number of rules (all that are needed by default).

    Returns
    -------
    OutcomeSummary
    """
    positive = np.asarray(positive, dtype=bool)
    allows = [allowed_rows(lmf.matrix[:, indices])
              for _, _, indices in lmf.groups]
    candidates, seen = [], set()
    for row in np.flatnonzero(positive):
        box = grow_rule(lmf, allows, positive, int(row))
        key = tuple(mask.tobytes() for mask in box)
        if key not in seen:
            seen.add(key)
            candidates.append((box, _contained(allows, box) & positive))
    uncovered = positive.copy()

    def score(index):
        """The priority of a candidate (smaller first): more new leaves,
        more of their samples, fewer conditions, earlier candidates."""
        box, contained = candidates[index]
        new = contained & uncovered
        weight = 0 if samples is None else int(samples[new].sum())
        restricted = sum(not mask.all() for mask in box)
        return (-int(new.sum()), -weight, restricted, index)

    # lazy greedy set cover: the scores of the candidates only get worse as
    # leaves get covered, so a stale score is a bound on the current one
    queue = [score(i) for i in range(len(candidates))]
    heapq.heapify(queue)
    rules = []
    while uncovered.any() and (max_rules is None or len(rules) < max_rules):
        current = score(heapq.heappop(queue)[-1])
        if queue and current > queue[0]:
            heapq.heappush(queue, current)
            continue
        box, contained = candidates[current[-1]]
        uncovered &= ~contained
        leaves = np.flatnonzero(contained)
        rule = Rule(outcome=outcome, conditions=_conditions(lmf, box),
                    leaves=lmf.leaf_ids[leaves].tolist())
        if samples is not None:
            rule.samples = int(samples[leaves].sum())
            if probabilities is not None and rule.samples:
                rule.purity = float((samples[leaves] * probabilities[leaves])
                                    .sum() / rule.samples)
        if values is not None:
            rule.values = (float(values[leaves].min()),
                           float(values[leaves].max()))
        rules.append(rule)
    return OutcomeSummary(outcome=outcome, rules=rules,
                          leaves=lmf.leaf_ids[positive].tolist(),
                          uncovered=lmf.leaf_ids[uncovered].tolist())
