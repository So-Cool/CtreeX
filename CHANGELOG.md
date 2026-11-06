# Changelog

## 0.0.4

* **Class-equivalent explanations.**
  `explain_instance(..., counterfactual=False)` returns the changes that move
  a data point to another leaf of the *same* class, i.e., that keep the
  prediction (e.g., "the loan would still be declined had the applicant
  asked for at most 5000");
  the default (`counterfactual=True`) remains the class-contrastive
  counterfactuals.
  The kind is recorded in `CounterfactualExplanation.counterfactual`.
  (The algorithm supported both kinds internally,
  as did the original implementation;
  they are now part of the public API.)

## 0.0.3

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

## 0.0.2

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

## 0.0.1

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

### Critical fixes (always applied)

* Errors were raised as strings (`raise('...')`), which raises `TypeError` in
  Python 3; specific exceptions with the intended message are raised now.
* `raise NotImplemented` (not an exception) is now `NotImplementedError`.
* With pandas >= 1 the support sets of the leaf-to-leaf distances raised
  `KeyError` for empty distance bins; they are empty lists again.
* The tree-representation helpers did not import (undefined names);
  they were rewritten.
* Debug output (feature frequencies printed on every call) is logged instead.

### Switchable fixes (`implementation='fixed'`, the default)

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
