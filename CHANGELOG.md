# Changelog

## 0.1.2 (unreleased)

* **Style checks.**
  The code and tests follow `flake8` (configured in `.flake8`), which is
  part of the development dependencies and runs in CI (the `lint` job).

## 0.1.1

* **Training statistics.**
  `TreeStructure` stores the number of training samples
  (`n_node_samples`), their total weight (`weighted_n_node_samples`) and the
  `impurity` of every node;
  `from_sklearn` fills them in and the JSON format (version 2) records them
  when present.
  They are optional, e.g., for hand-built trees and older files.
  `TargetContext` gains `leaf_samples` and `leaf_impurity`.
* **Combinable targets.**
  Targets combine with `&` (`AllOf`), `|` (`AnyOf`) and `~` (`Not`);
  e.g., `OtherClass() & MinSamples(20)`.
* **Leaves backed by enough data.**
  `MinSamples(n)` selects leaves with at least `n` training samples.
* **Ranking options.**
  `explain_instance(..., rank_by=...)` orders the counterfactuals at every
  distance by feature frequency (`'frequency'`, the default and the previous
  behaviour), the number of training samples of their leaves (`'samples'`,
  the total for merged counterfactuals) or the purity of their leaves
  (`'purity'`, the majority proportion, or the lowest impurity
  for regression trees, of the least pure merged leaf);
  ties keep the frequency order.
  `rank_counterfactuals` takes the corresponding `leaf_scores` and
  `aggregate`.
* **Supportive explanations.**
  `TreeCounterfactualExplainer.supportive_explanations(x, target)` finds all
  the smallest sets of features of a data point that guarantee its
  prediction (`SameClass()` by default, a target such as `ValueWithin(10)`
  for regression trees) whatever the values of the other features;
  every feature is fixed at the meta-feature of the data point's value, i.e,,
  the interval between the two thresholds around it or its category.
  The search considers all the features used by the tree, so explanations
  may include features that the path of the data point does not test.
* **Invariance ranges.**
  `TreeCounterfactualExplainer.invariance_ranges(x, target)` lists the values
  of every feature used by the tree that keep the prediction when the other
  features do not change.
* **Contextualised counterfactuals.**
  `TreeCounterfactualExplainer.contextualise(explanation, max_size)` adds to
  every ranked counterfactual all the smallest sets of the other features
  (fixed at the data point's values) that guarantee its outcome whatever the
  values of the remaining features;
  `max_size` (the depth of the tree by default) limits the search and a
  counterfactual without a context of at most `max_size` features raises
  `ValueError`.
* **New supportive explanation module.**
  New `ctreex.supportive` module with the `SupportiveExplanation` and
  `ContextualisedCounterfactual` dataclasses and the functions behind the
  three methods;
  their conditions use the counterfactual format so `describe` works.
  `describe_range` describes an interval without bounds as `'any value'`
  (instead of raising `ValueError`).
* **Ordered categorical features.**
  `Feature(..., ordered=True)` marks ordinal categories;
  the `values` given with an ordered feature set the order of its categories
  (otherwise the order of its columns is used).
  Numerical features cannot be marked as ordered (they always are).
* **Direction constraints.**
  `explain_instance(..., directions={'age': 'increase'})` lets numerical and
  ordered categorical features change only in the given direction;
  the values in the other direction are removed from the counterfactuals and
  a leaf is used only if some of its values remain
  (`ctreex.counterfactuals.in_direction`, `extract_counterfactuals` takes
  `directions`).
* **Monotonicity audits.**
  `TreeCounterfactualExplainer.monotonicity_violations(feature, direction, positive)`
  lists the pairs of leaves that differ only on the feature where changing it
  in the direction makes the prediction worse, i.e., moves away from the
  `positive` class or, for regression trees, lowers the prediction;
  every `MonotonicityViolation` holds the two leaves and their predictions,
  the values of the feature `before` and `after` the change as well as the
  `context` of the other features shared by the two leaves
  (in the counterfactual format).
  New module `ctreex.audit`.
* **Feature audits.**
  `TreeCounterfactualExplainer.feature_audit(feature, X, target, tolerance)`
  shows where a feature alone changes the prediction, e.g., to check whether
  a protected attribute decides the outcome.
  The `FeatureAudit` lists all the pairs of leaves that differ only on the
  feature and predict different outcomes (`FeatureFlip`, with the values of
  the feature `before` and `after` and the shared `context`,
  `ctreex.audit.feature_flips`) and the leaves whose prediction does
  (`affected_leaves`) or does not (`unaffected_leaves`) depend on the
  feature.
  With data `X`, it also lists the indices of the data points whose
  prediction changes when the feature alone changes (`alone`) and of those
  whose closest counterfactuals involve the feature (`closest`).
  Regression trees need a `tolerance` (leaf predictions that differ by more
  than it are different) and, with data, a `target`.
* **Instance-contrastive explanations.**
  `TreeCounterfactualExplainer.explain_contrast(x, x_reference, same_leaf, tolerance, **kwargs)`
  explains how `x` would have to change to be treated like `x_reference`:
  the counterfactuals lead to the class of the reference (`TargetClass`),
  to the leaves predicting at most `tolerance` more or less than it
  (regression trees, `tolerance` required) or, with `same_leaf=True`, to its
  leaf;
  at every distance, those whose new values include the values of the
  reference come first and `DistanceCounterfactuals.reference_matches` lists
  these features.
  `CounterfactualExplanation.reference` holds the values of the reference.
  Data points with the same outcome (without `same_leaf`) or the same leaf
  raise `ValueError`;
  the other arguments of `explain_instance` (except `target`) are passed on.
* **Leaf targets.**
  `Leaves(*ids)` selects given leaves.
* **Data points of counterfactuals.**
  New `ctreex.points` module.
  `TreeCounterfactualExplainer.counterfactual_point(x, counterfactual, how)`
  returns the data point of a counterfactual nearest to `x` (`how='nearest'`)
  or in the middle of its ranges (`how='midpoint'`) and
  `sample_counterfactual(x, counterfactual, n, random_state)` draws data
  points uniformly at random from its ranges (proportionally to their
  lengths) and categories, with a seed;
  e.g., to share a counterfactual without revealing the thresholds of the tree.
  The features the counterfactual does not change keep the values of `x`;
  both accept contextualised counterfactuals and check that every data point
  reaches a leaf of the counterfactual.
  Numerical values are 32-bit floating point numbers (whole numbers for
  integer features), so scikit-learn and `TreeStructure` route them alike.
* **Feature ranges and steps.**
  `Feature` has optional `minimum` and `maximum` (bounding unbounded ranges
  when sampling or taking their middle) and `step` (the smallest change of
  a real-valued feature used by the nearest data point);
  `FeatureSpec.fit_ranges(X)` sets the ranges of the numerical features from
  data.
* **Hidden thresholds.**
  `describe(counterfactuals, hide_thresholds=True, random_state=...)`
  replaces the ranges of numerical features with random values from them.
* **Exemplars.**
  `TreeCounterfactualExplainer.counterfactual_leaves(x, counterfactual)`
  lists the leaves a counterfactual leads to
  (`ctreex.supportive.counterfactual_leaves`) and
  `exemplars(X, leaves, n, random_state)` returns the data points of `X` in
  given leaves (at most `n`, drawn at random).
* **Decoding.**
  `TreeCounterfactualExplainer.decode(X)` returns the base feature values of
  one data point or a list of them.
* **Plain-text explanations.**
  New `ctreex.text` module with a generic template built from the display
  names of the features, the names of their categories and the names of the
  classes (no templates per data set).
  `TreeCounterfactualExplainer.explanation_text(explanation, contextualised, n, integer_ranges, hide_thresholds, random_state)`
  returns the prediction followed by one sentence per counterfactual, e.g.,
  "Had duration been less or equal to 22.5 (instead of 48), the prediction
  would have been low risk.", with ", provided that ..." for contextualised
  counterfactuals;
  `supportive_text(x, target, integer_ranges)` verbalises supportive
  explanations.
* **Notebooks.**
  New `04_beyond_counterfactuals.ipynb` demonstrates the additions of this
  version on the German Credit data:
  training statistics in targets and rankings, supportive explanations,
  invariance ranges and contextualised counterfactuals, direction constraints
  and monotonicity audits, feature audits, instance-contrastive
  explanations, data points of counterfactuals (nearest, sampled, hidden
  thresholds, exemplars) and plain-text explanations.

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
