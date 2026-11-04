# CtreeX

> Counterfactual decision tree explainer.

For a data point classified by a decision tree, CtreeX finds the smallest
changes of its features that would lead to a different prediction;
for example, "had you asked for between £955 and £1,181 instead of £3,959,
your loan would have been accepted".
It depends only on `numpy`.

## Installation

Python 3.9 or newer is required.

```sh
pip install "ctreex @ git+https://github.com/So-Cool/ctreex.git@v0.0.1"
```

or with the scikit-learn helpers and the plots

```sh
pip install "ctreex[sklearn,plot] @ git+https://github.com/So-Cool/ctreex.git@v0.0.1"
```

## Quick start

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ctreex import Feature, FeatureSpec, TreeCounterfactualExplainer

# a numerical feature (income) and a categorical one (housing), one-hot encoded
spec = FeatureSpec(
    [('income', None),
     ('housing', 'rent'),
     ('housing', 'own')],
    features=[Feature('housing', 'categorical',
                      value_names={'rent': 'renting', 'own': 'owning a house'})])
X = np.array([[1000, 1, 0], [3000, 1, 0], [1500, 0, 1], [1500, 1, 0],
              [800, 0, 1]] * 5)
y = np.array(['declined', 'accepted', 'accepted', 'declined', 'declined'] * 5)
clf = DecisionTreeClassifier(random_state=0).fit(X, y)

explainer = TreeCounterfactualExplainer(clf, spec)
explanation = explainer.explain_instance(spec.encode({'income': 1500, 'housing': 'rent'}))
explanation.prediction                     # 'declined'
for distance, counterfactual in explanation.ranked():
    print(distance, explainer.describe({0: counterfactual})[0])
# 1 {'income':  {'is': [1500],      'wannabe': ['more than 2250.0']}}
# 1 {'housing': {'is': ['renting'], 'wannabe': ['owning a house']}}
```

`notebooks/` has two worked examples:
a quick start and the [UCI German Credit][UCIgc] data.

[UCIgc]: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## How it works

1. **Meta-features.**
   The conditions on the root-to-leaf paths become boolean meta-features:
   the one-hot columns used by the tree and
   the intervals between the split thresholds of every numerical feature
   (e.g., `age<=30.0`, `30.0<age<=50.0` and `50.0<age`).
   Every leaf is a vector over the meta-features with values
   *1* for implied by its path, *-1* for contradicted and
   *0* for unconstrained (`ctreex.meta_features`).
2. **Distances.**
   A data point is described with the same meta-features.
   Its *exact* distance to a leaf is the number of features whose values
   would have to change to reach the leaf (`ctreex.distances`).
3. **Counterfactuals.**
   The leaves of another class at distance 1, 2, ... yield the counterfactuals.
   At distance 1 they are merged per feature (adjacent intervals are joined);
   features used by more leaves come first (`ctreex.counterfactuals`).

Counterfactuals can be restricted:

* `excluded=['age']` -- never change these features; and
* `included=['income']` -- change only these features.

The leaf-to-leaf distance matrix (`explainer.leaf_distance_matrix()`), its
heat map and histograms (`ctreex.plot`) and the *volatility index* of a leaf
-- i.e., how close it is to leaves of other classes
(`ctreex.counterfactuals.volatility_index`) -- describe the structure of the
tree as a whole.

## API

| Object | Purpose |
|---|---|
| `TreeCounterfactualExplainer(model, feature_spec, implementation, fixes, class_names)` | the explainer; `explain_instance(x, instance_values, max_distance, excluded, included)` returns a `CounterfactualExplanation` |
| `CounterfactualExplanation` | `prediction`, `leaf`, `leaf_distances`, `support` (leaves per distance), `counterfactuals` (one `DistanceCounterfactuals` per distance), `ranked()`, `at_distance(d)` |
| `FeatureSpec`, `Feature` | maps the columns of a one-hot encoded array to features; display names and category names; `encode`/`decode` |
| `TreeStructure`, `load_tree` | a fitted tree independent of scikit-learn: from scikit-learn, to/from JSON, from pickles |
| `ctreex.sklearn_tools` | Graphviz, text and matplotlib representations of scikit-learn trees, root-to-leaf paths |
| `ctreex.plot` | heat map, bar plot and histogram of leaf distances (`plot` extra) |

Trees are stored as JSON with `TreeStructure.from_sklearn(clf).to_json(path)`,
since scikit-learn pickles only load in the version that wrote them.

## Legacy and fixed behaviour

* `implementation='legacy'` reproduces the original (2018) implementation of the
algorithm;
* `implementation='fixed'` (default) enables the fixes listed in
  the [changelog](#001) section below.

It is possible to enable specific fixes for individual implementations too;
e.g.,
```python
TreeCounterfactualExplainer(clf, spec, implementation='legacy',
                            fixes={'deterministic_order': True})
```

## Development

```sh
uv sync                     # Python 3.12 environment with the test dependencies
uv run pytest
uv sync --group notebooks && uv run jupyter lab notebooks/
```

## Changelog

### 0.0.1

A numpy reimplementation of the original (2018) CtreeX code,
which was based on pandas and seaborn.

* **Equivalence.**
  With `implementation='legacy'` the explainer reproduces
  the original implementation (meta-features, leaf descriptions,
  leaf-to-leaf distances, volatility indices and
  the counterfactuals of 50 data points at every distance.
* **API.**
  `TreeCounterfactualExplainer`,
  `FeatureSpec` (explicit description of one-hot encoded features;
  the original parsed `<feature>_<category>` column names,
  so feature names could not contain `_`),
  `TreeStructure` (version-independent JSON storage of trees), 
  `ctreex.sklearn_tools` (tree representations) and
  `ctreex.plot`.
* **Speed.**
  The leaf-to-leaf distance matrix is about 30 times faster.

#### Critical fixes (always applied)

* Errors were raised as strings (`raise('...')`), which raises `TypeError` in
  Python 3; specific exceptions with the intended message are raised now.
* `raise NotImplemented` (not an exception) is now `NotImplementedError`.
* With pandas >= 1 the support sets of the leaf-to-leaf distances raised
  `KeyError` for empty distance bins; they are empty lists again.
* The tree-representation helpers did not import (undefined names);
  they were rewritten.
* Debug output (feature frequencies printed on every call) is logged instead.

#### Switchable fixes (`implementation='fixed'`, the default)

| Switch | Legacy behaviour | Fixed behaviour |
|---|---|---|
| `deterministic_order` | alternative categories de-duplicated with `list(set(...))`, so their order changed between program runs (string hashing is randomised) | order of discovery |
| `continue_past_missing_distance` | the search stopped at the first distance at which no leaf exists, e.g. with leaves at distances 1 and 3 only the distance-3 counterfactuals were never found | missing distances are skipped |

Select individual fixes with `fixes={'deterministic_order': False}` or a
`ctreex.Fixes` object.

#### Kept from the original

* A numerical feature split at a single threshold `t` is described only by the
  `A<=t` meta-feature; counterfactuals for the complementary interval are
  produced by swapping the interval bounds (they are tuples rather than lists
  and are therefore de-duplicated like categories when ranked).
* The first (last) interval of a numerical feature is set for a leaf when its
  path has no lower (upper) bound on the feature.
