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
pip install "ctreex @ git+https://github.com/So-Cool/ctreex.git@v0.0.3"
```

or with the scikit-learn helpers and the plots

```sh
pip install "ctreex[sklearn,plot] @ git+https://github.com/So-Cool/ctreex.git@v0.0.3"
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

`notebooks/` has three worked examples:
a quick start, the [UCI German Credit][UCIgc] data and a tour of the rest of
the API (feature specifications, explanation details, individual fixes, tree
storage, leaf-distance plots and the scikit-learn helpers).

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

For multi-class trees, `target_class='...'` restricts the counterfactuals to
one class, and `explanation.ranked(with_classes=True)` reports the class each
counterfactual leads to.

The leaf-to-leaf distance matrix (`explainer.leaf_distance_matrix()`), its
heat map and histograms (`ctreex.plot`) and the *volatility index* of a leaf
-- i.e., how close it is to leaves of other classes
(`ctreex.counterfactuals.volatility_index`) -- describe the structure of the
tree as a whole.

## API

| Object | Purpose |
|---|---|
| `TreeCounterfactualExplainer(model, feature_spec, implementation, fixes, class_names)` | the explainer; `explain_instance(x, instance_values, max_distance, excluded, included, target_class)` returns a `CounterfactualExplanation` |
| `CounterfactualExplanation` | `prediction`, `leaf`, `leaf_distances`, `support` (leaves per distance), `counterfactuals` (one `DistanceCounterfactuals` per distance), `ranked()`, `at_distance(d)` |
| `FeatureSpec`, `Feature` | maps the columns of a one-hot encoded array to features; display names and category names; `encode`/`decode` |
| `TreeStructure`, `load_tree` | a fitted tree independent of scikit-learn: from scikit-learn, to/from JSON, from pickles |
| `ctreex.sklearn_tools` | Graphviz, text and matplotlib representations of scikit-learn trees, root-to-leaf paths |
| `ctreex.plot` | heat map, bar plot and histogram of leaf distances (`plot` extra) |

Trees are stored as JSON with `TreeStructure.from_sklearn(clf).to_json(path)`,
since scikit-learn pickles only load in the version that wrote them.

Supported models are binary decision tree *classifiers* (binary and
multi-class); regression trees and multi-output trees are not supported.
Missing values (`nan`) are routed as learnt by scikit-learn (1.3 or newer)
when predicting, but data points with missing values cannot be explained.
The `value` array of a `TreeStructure` holds class counts or -- for trees
built with scikit-learn 1.4 or newer -- class proportions; only their relative
sizes are used.
When a tree stores feature names and they differ from the column names of the
feature specification, the explainer warns.

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

### 0.0.3

* **Multi-class trees.**
  `explain_instance(..., target_class=...)` explains how to reach a given
  class; by default counterfactuals lead to any other class.
  Every `DistanceCounterfactuals` records the class of each leaf
  (`leaf_classes`) and the classes each ranked counterfactual leads to
  (`ranked_classes`); `explanation.ranked(with_classes=True)` returns
  `(distance, classes, counterfactual)` triples.
* **New fix: distinguish counterfactuals of different classes.**
  `per_class_ranking` (enabled by `implementation='fixed'`):
  at distance 1 counterfactuals are merged and scored per class.
  The legacy implementation merged the counterfactuals of a feature across
  leaves of *different* classes, so a merged range (e.g., "more than 9.5")
  could lead to different classes, which the explanation did not show.
  Two-class trees are not affected.
* **New fix: display shorter threshold descriptions.**
  `short_thresholds` (enabled by `implementation='fixed'`):
  `describe` shows thresholds in their shortest equivalent form.
  scikit-learn computes the thresholds of 32-bit data, which were described
  in full, e.g., `'more than 0.800000011920929'`;
  the largest 32-bit number not exceeding a threshold selects the same data,
  and its shortest form is used when it is shorter (`'more than 0.8'`;
  `ctreex.counterfactuals.format_bound`).
  Only the descriptions change; thresholds such as `954.5` or `0.1` are kept.

### 0.0.2

* **New fix: two numerical intervals.**
  `complete_numerical_intervals` (enabled by `implementation='fixed'`):
  a numerical feature split at a single threshold `t` is now described by two
  meta-features, `A<=t` and `t<A`, like features with more thresholds;
  the legacy implementation created only `A<=t` and derived "more than `t`"
  by swapping the bounds of the excluded interval.
  The counterfactuals of data points and their distances to the leaves are
  unchanged;
  the leaf-to-leaf distances -- hence the volatility index and the leaf
  support sets -- change since every interval contributes to them.
* **New fix: feature exclusion and inclusion.**
  `whole_counterfactual_restrictions` (enabled by `implementation='fixed'`):
  the `excluded` and `included` features of `explain_instance` select whole
  counterfactuals, i.e., a leaf is used only if all the features that have to
  change to reach it are allowed.
  The legacy implementation removed the forbidden features from the
  counterfactual of every leaf instead, which produced incomplete
  counterfactuals (e.g., a distance-2 counterfactual changing a single
  feature, which does not change the prediction) and empty ones.
  With the fix every counterfactual at distance `d` changes exactly `d`
  features.
* **New fix: pay attention of all values of categorical features.**
  `complete_categorical_features` (enabled by `implementation='fixed'`):
  meta-features are created for *all* the categories of every categorical
  feature used by the tree, not only for the categories the tree splits on.
  The legacy implementation completed a leaf's description with the rule
  "all but one category excluded, hence the remaining one is required",
  which is wrong when the tree never splits on some categories: data points
  could be at a non-zero distance from their own leaf, counterfactuals got
  wrong distances or were missed, and some alternative categories were not
  listed. This was also the only cause of distances without leaves observed in
  practice, hence `continue_past_missing_distance` is now merely a safety net.
* **Missing values.**
  `TreeStructure` stores where missing values (`nan`) go at every split
  (`missing_go_to_left`, learnt by scikit-learn 1.3 or newer; also saved in
  JSON) and follows it when predicting; previously missing values always went
  to the right child, which could disagree with scikit-learn.
  `explain_instance` rejects data points with missing values with a clear
  error (counterfactuals of missing values are not defined yet).
* **Whole numbers.**
  `Feature(..., integer=True)` marks numerical features that take whole
  values; `describe(..., integer_ranges=True)` then describes their ranges
  with whole numbers, e.g., `'between 955 and 1181'` instead of
  `'between 954.5 and 1181.5'` (`ctreex.counterfactuals.describe_integer_range`).
* **Documentation.**
  How leaves describe numerical features is documented more accurately:
  an interval is `1` when it lies inside the region the leaf's path allows
  for the feature and `-1` otherwise (e.g., the first interval is `1` exactly
  when the path has no lower bound); all intervals are `0` when the path does
  not test the feature.
* **Notebook.**
  `03_api_tour.ipynb` demonstrates the rest of the API:
  `FeatureSpec.from_column_names` and `decode`, class names, `predict_leaf`,
  `max_distance`, `at_distance`, per-leaf and per-feature counterfactuals,
  individual fixes, `TreeStructure` (`apply`, `predict_proba`, JSON and
  pickles), `leaf_metric_barplot`, `support_set_metric` and all
  `ctreex.sklearn_tools` helpers.

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
| `complete_numerical_intervals` (0.0.2) | a numerical feature split at a single threshold `t` had only the `A<=t` meta-feature | both `A<=t` and `t<A` |
| `complete_categorical_features` (0.0.2) | meta-features only for the categories the tree splits on; leaves could require a category although others were allowed | meta-features for all the categories of the features used by the tree |
| `short_thresholds` (0.0.3) | thresholds of 32-bit data described in full (`0.800000011920929`) | shortest equivalent form (`0.8`) |
| `per_class_ranking` (0.0.3) | distance-1 counterfactuals of a feature were merged across leaves of different classes (multi-class trees) | merged and scored per class |
| `whole_counterfactual_restrictions` (0.0.2) | `excluded`/`included` features were removed from every counterfactual, leaving incomplete and empty counterfactuals | leaves whose counterfactual needs a forbidden feature are skipped |

Select individual fixes with `fixes={'deterministic_order': False}` or a
`ctreex.Fixes` object.
