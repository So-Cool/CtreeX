# Changelog

## 0.1.0

* **The legacy implementation is removed.**
  The explainer has a single behaviour -- the one previously selected with
  `implementation='fixed'` (the default) -- so
  the `implementation` and `fixes` arguments of `TreeCounterfactualExplainer`,
  its `fixes` and `implementation` attributes, and
  `ctreex.Fixes` (the `ctreex.fixes` module) are gone.
  All the fixes listed under [0.0.1](#001) are always applied:
  categorical features are described by all their categories,
  numerical features split at a single threshold by two intervals,
  `excluded` and `included` select whole counterfactuals,
  distance-1 counterfactuals are merged per class,
  alternative values keep the order in which they were found,
  distances without leaves are skipped and
  `describe` shows thresholds in their shortest form
  (the structured counterfactuals keep the exact thresholds).
  To reproduce the original implementation, use version 0.0.4.
* **Legacy behaviour has changed.**
  Functions whose arguments only selected the legacy behaviour changed:
  `build_meta_features(splits, spec)`, `extract_counterfactuals`
  (without `spec` and `whole_restrictions`),
  `rank_counterfactuals` (without `deterministic_order`) and
  `describe_counterfactuals` (without `short_thresholds`).
* **New leaf-to-leaf distance.**
  The distance between two leaves is now the number of base features on
  which they are incompatible, i.e., allow no common interval (category);
  e.g., sibling leaves are at distance 1.
  This mirrors the distance between a data point and a leaf
  (the number of features that have to change).
  Previously numerical features contributed the number of disagreeing
  intervals divided by the total number of meta-features -- so distances
  changed when unrelated features gained thresholds -- and categorical
  features the fraction of disagreeing completions.
  `leaf_distance_matrix` returns whole numbers (an integer array);
  `ctreex.distances.allowed` and `ctreex.distances.incompatible` replace
  `numerical_identity`, `numerical_distance` and `categorical_distance`.
* **Volatility index and leaf support sets use whole-number distances.**
  `volatility_index` is defined as the sum over the distances `n = 1, ..., N`
  (the largest distance in the matrix) of `(other + 1) / (n * (same + 1))`,
  where `other` (`same`) is the number of leaves of another (the same) class at
  distance `n`;
  previously the distances were binned (`bins_step`) and the leaf itself
  counted towards the first bin.
  `support_set_metric` returns `{distance: {'same': [...], 'other': [...]}}`
  for every distance from 0 (the leaf itself) instead of interval labels and
  both functions -- as well as `ctreex.plot.leaf_metric_histogram` -- lose
  the `bins_step` argument;
  the two functions no longer bin the distances differently.
* **Merged ranges are ordered correctly.**
  At distance 1 the merged ranges of a numerical feature are listed by
  their lower bounds with the unbounded one (`less or equal to ...`) first;
  previously it was sorted as if its lower bound was 0, which misplaced it when
  the thresholds were negative.
* **Feature restrictions are validated.**
  `excluded` and `included` have to name features of the specification;
  unknown names (e.g., typos or display names) raise `ValueError` and a
  single string raises `TypeError` (previously unknown names were silently
  ignored and a string was matched as a substring).
* **Targets.**
  The leaves the counterfactuals lead to are selected by a *target*
  (`ctreex.targets`), given with `explain_instance(..., target=...)`:
  `OtherClass()` (the default, class-contrastive counterfactuals),
  `TargetClass(label)` (one class of a multi-class tree) and
  `SameClass()` (class-equivalent explanations).
  They replace the `target_class` and `counterfactual` arguments of
  `explain_instance`:
  `target_class='a'` is now `target=TargetClass('a')` and
  `counterfactual=False` is `target=SameClass()`.
  `CounterfactualExplanation.target` records the target
  (instead of the `counterfactual` flag).
* **Custom targets.**
  Subclasses of `ctreex.Target` implement `select(context)`, which returns a
  boolean mask over the leaves, and optionally `validate(context)`;
  the `ctreex.targets.TargetContext` holds the prediction, the leaf of the
  data point, the leaves with their classes and class proportions, and the
  classes.
  `TreeCounterfactualExplainer.target_context(leaf)` builds it.
* **Support sets of data points are split by the target.**
  `CounterfactualExplanation.support` (`ctreex.counterfactuals.support_set`)
  maps every distance to `{'target': [...], 'non_target': [...]}` instead of
  `{'same': [...], 'other': [...]}`;
  `support_set(distances, leaf_ids, targets)` takes the boolean mask of the
  target leaves, and `extract_counterfactuals` uses its `target` leaves
  (without the `counterfactual` and `target_class` arguments).
  The support sets of leaves (`support_set_metric`) keep `same` and `other`,
  which refer to the class of the reference leaf.
* **Regression trees.**
  `TreeStructure` describes regression trees (without `classes`, a single
  value per node), e.g., built from `sklearn.tree.DecisionTreeRegressor`;
  its `task` is `'regression'` (`'classification'` otherwise),
  `predict` returns values, `leaf_value` gives the prediction of a leaf and
  `predict_proba` and `leaf_class` raise `TypeError`.
  The JSON format (version 2) records the `task`;
  version 1 files are read as classifiers.
* **Targets of regression trees.**
  Regression trees need a target:
  `ValueChange(delta)` for leaves predicting at least `prediction + delta` for
  a positive `delta` and at most `prediction + delta` for a negative one; and
  `ValueAtLeast(value)`, `ValueAtMost(value)` and
  `ValueWithin(tolerance)` for the changes that keep the prediction within the
  tolerance.
  At distance 1 only the counterfactuals of leaves predicting the same value
  are merged.
  `DistanceCounterfactuals.leaf_values` and `ranked_values` report the
  values of the leaves and the range of the values every ranked
  counterfactual leads to (instead of `leaf_classes` and `ranked_classes`),
  `ranked(with_values=True)` returns
  `(distance, (smallest, largest), counterfactual)` triples and
  `CounterfactualExplanation.task` records the kind of the tree.
  The `TargetContext` has the `task` and the `leaf_values`
  (its class attributes are `None` for regression trees).
* **Leaf comparisons of regression trees.**
  `support_set_metric`, `volatility_index`, `leaf_metric_barplot` and
  `leaf_metric_histogram` take a `tolerance`:
  leaves whose values differ by at most the tolerance are the same
  (`ctreex.counterfactuals.same_leaves`).
  `TreeCounterfactualExplainer.support_set_metric(leaf, tolerance)` and
  `volatility_index(leaf, tolerance)` require it for regression trees
  (and reject it for classification trees);
  `leaf_distance_matrix` returns the values of the leaves instead of their
  classes.
* **Several target classes.**
  `TargetClass('a', 'b')` leads to class `a` or `b`;
  the classes are stored in `TargetClass.labels` (instead of `label`) and
  none of them can be the prediction.
* **Confident leaves.**
  `ClassProbability('a', ..., at_least=p)` leads to leaves that predict one
  of the classes with a proportion of at least `p`;
  the classes may include the prediction.
* **Numerical columns described as categories.**
  Column names containing the separator (e.g., `petal_length`) are read as
  one-hot columns (`<feature>_<category>`) by
  `FeatureSpec.from_column_names` unless they are listed as `numerical`;
  the explainer now detects one-hot columns split at thresholds outside of
  `[0, 1)` and explains the likely cause (previously the error was, e.g.,
  "One-hot error: too many 1s for feature 'petal'.").
* **Notebooks.**
  The notebooks no longer use the legacy implementation and `03_api_tour.ipynb`
  demonstrates several target classes, `ClassProbability`, the
  `TargetContext` and a custom target, and regression trees
  (the regression targets, `ranked(with_values=True)`, leaf comparisons with
  a tolerance and JSON storage) on the diabetes data shipped with scikit-learn.

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
