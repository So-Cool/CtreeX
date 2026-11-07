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
pip install "ctreex @ git+https://github.com/So-Cool/ctreex.git@v0.0.4"
```

or with the scikit-learn helpers and the plots

```sh
pip install "ctreex[sklearn,plot] @ git+https://github.com/So-Cool/ctreex.git@v0.0.4"
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
the API (feature specifications, explanation details, tree storage,
leaf-distance plots and the scikit-learn helpers).

[UCIgc]: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## How it works

1. **Meta-features.**
   The conditions on the root-to-leaf paths become boolean meta-features:
   all the categories (one-hot columns) of every categorical feature used by
   the tree and
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

The names are those of the features of the specification
(`FeatureSpec.features`), not their display names;
unknown names raise an error.

A leaf is used only if all the features that have to change to reach it are
allowed, so every counterfactual at distance *d* changes exactly *d* features.

For multi-class trees, `target_class='...'` restricts the counterfactuals to
one class, and `explanation.ranked(with_classes=True)` reports the class each
counterfactual leads to.

With `counterfactual=False`, `explain_instance` returns *class-equivalent*
explanations instead:
the changes that keep the prediction (they lead to other leaves of the same
class), e.g., to show how much a feature can change without affecting
the decision.

The leaf-to-leaf distance matrix (`explainer.leaf_distance_matrix()`), its
heat map and histograms (`ctreex.plot`) and the *volatility index* of a leaf
-- i.e., how close it is to leaves of other classes
(`ctreex.counterfactuals.volatility_index`) -- describe the structure of the
tree as a whole.
The distance between two leaves is the number of features on which they
allow no common value (their intervals or categories do not overlap).
For a leaf with `other` (`same`) leaves of another (the same) class at
distance *n*, the volatility index sums `(other + 1) / (n * (same + 1))`
over the distances *n = 1, 2, ...*;
large values indicate leaves close to other classes.

## API

| Object | Purpose |
|---|---|
| `TreeCounterfactualExplainer(model, feature_spec, class_names)` | the explainer; `explain_instance(x, instance_values, max_distance, excluded, included, target_class, counterfactual)` returns a `CounterfactualExplanation` |
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

## Development

```sh
uv sync                     # Python 3.12 environment with the test dependencies
uv run pytest
uv sync --group notebooks && uv run jupyter lab notebooks/
```

## The original implementation

CtreeX started as a pandas-based implementation written in 2018.
Versions 0.0.1 to 0.0.4 could reproduce it (`implementation='legacy'`) and
switch its fixes on and off individually;
since 0.1.0 only the fixed behaviour is available.
To reproduce the original results, install version 0.0.4
(see the [changelog](./CHANGELOG.md#001) for the list of fixes).

## Changelog

The [`CHANGELOG.md`](./CHANGELOG.md) file contains a detailed changelog.
