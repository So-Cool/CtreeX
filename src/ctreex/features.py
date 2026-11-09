"""
Description of the (one-hot encoded) features a decision tree was trained on.

The explainer reasons about *base* features -- e.g., ``age`` (numerical) or
``housing`` (categorical) -- while the tree splits on *columns* of a one-hot
encoded data array -- e.g., ``age``, ``housing_rent``, ``housing_own``.
A :class:`FeatureSpec` maps every column to its base feature (and category).
"""
import dataclasses

from ctreex import _validation as val

__all__ = ['Feature', 'FeatureSpec']

NUMERICAL = 'numerical'
CATEGORICAL = 'categorical'


@dataclasses.dataclass
class Feature(object):
    """
    A base feature.

    Attributes
    ----------
    name : str
        Identifier of the feature (e.g., ``A13``).
    kind : {'numerical', 'categorical'}
        Feature type.
    values : list of str
        Categories (for categorical features) in the order of their one-hot
        columns, or, for ordered features given with their values, in
        their order; empty for numerical features.
    display_name : str
        Human-readable name (defaults to ``name``).
    value_names : dict
        Human-readable names of the categories (defaults to the categories).
    integer : bool
        Whether a numerical feature takes only whole values; its ranges can
        then be described with whole numbers (see
        :func:`ctreex.counterfactuals.describe_counterfactuals`).
    ordered : bool
        Whether the categories of a categorical feature are ordered (ordinal),
        e.g., years of employment;
        their order is the order of ``values``, which give the values with the
        feature to set it, otherwise the order of the one-hot columns is used.
        Ordered features can be restricted to change in one direction
        (``explain_instance(..., directions=...)``) and audited for
        monotonicity.
        Numerical features are always ordered.
    """
    name: str
    kind: str
    values: list = dataclasses.field(default_factory=list)
    display_name: str = None
    value_names: dict = dataclasses.field(default_factory=dict)
    integer: bool = False
    ordered: bool = False

    def __post_init__(self):
        if self.kind not in (NUMERICAL, CATEGORICAL):
            raise ValueError('Unknown feature kind: {!r}.'.format(self.kind))
        if self.integer and self.kind != NUMERICAL:
            raise ValueError('Only numerical features can be integers.')
        if self.ordered and self.kind != CATEGORICAL:
            raise ValueError('Only categorical features can be ordered '
                             '(numerical features always are).')
        if self.display_name is None:
            self.display_name = self.name

    def describe_value(self, value):
        """Returns the human-readable version of a category."""
        return self.value_names.get(value, value)


class FeatureSpec(object):
    """
    Maps the columns of a one-hot encoded data array to base features.

    Parameters
    ----------
    columns : list of tuple
        One ``(feature name, category)`` pair per column of the data array;
        ``category`` is ``None`` for numerical columns.
    features : list of Feature, optional
        Feature metadata (display names, value names). Features missing from
        this list are created automatically.
        The ``values`` of an ordered feature set the order of its categories
        (they have to be its categories); other features list their
        categories in the order of their columns.
    separator : str
        Used to build column names (``<feature><separator><category>``).

    Attributes
    ----------
    column_names : list of str
        Names of the columns (e.g., ``A1_3``); these are also the names of the
        categorical meta-features.
    features : dict
        Base features (``name -> Feature``) in the order of their first column.
    """

    def __init__(self, columns, features=None, separator='_'):
        val.check_columns(columns)
        self.columns = [(str(f), None if c is None else str(c))
                        for f, c in columns]
        self.separator = separator
        given = {f.name: f for f in (features or [])}
        self.features = {}
        for name, category in self.columns:
            kind = NUMERICAL if category is None else CATEGORICAL
            if name not in self.features:
                feature = given.get(name, Feature(name, kind))
                if feature.kind != kind:
                    raise ValueError('Feature {!r} has inconsistent '
                                     'columns.'.format(name))
                feature = dataclasses.replace(feature, values=[])
                self.features[name] = feature
            elif self.features[name].kind != kind:
                raise ValueError('Feature {!r} mixes numerical and '
                                 'categorical columns.'.format(name))
            if category is not None:
                if category in self.features[name].values:
                    raise ValueError('Duplicated column: {}={}.'.format(
                        name, category))
                self.features[name].values.append(category)
        for name, feature in self.features.items():
            order = given[name].values if name in given else []
            if feature.ordered and order:
                if sorted(map(str, order)) != sorted(feature.values):
                    raise ValueError(
                        'The values of the ordered feature {!r} have to be '
                        'its categories: {}.'.format(name, feature.values))
                feature.values = [str(v) for v in order]
        self.column_names = [
            name if category is None else
            '{}{}{}'.format(name, separator, category)
            for name, category in self.columns]

    @classmethod
    def from_column_names(cls, column_names, numerical=None, separator='_',
                          features=None):
        """
        Builds a specification from ``<feature><separator><category>`` names.

        Parameters
        ----------
        column_names : list of str
            Column names, e.g., ``['A2', 'A1_1', 'A1_2']``.
        numerical : list of str, optional
            Names of numerical columns. By default a column is categorical if
            (and only if) its name contains the ``separator``.
        """
        columns = []
        for name in column_names:
            is_numerical = (name in numerical if numerical is not None
                            else separator not in name)
            if is_numerical:
                columns.append((name, None))
            else:
                if separator not in name:
                    raise ValueError('Categorical column {!r} does not '
                                     'contain the separator.'.format(name))
                base, category = name.split(separator, 1)
                columns.append((base, category))
        return cls(columns, features=features, separator=separator)

    @property
    def n_columns(self):
        return len(self.columns)

    @property
    def n_features(self):
        return len(self.features)

    def kind(self, feature):
        return self.features[feature].kind

    def column_index(self, column_name):
        return self.column_names.index(column_name)

    def decode(self, row):
        """
        Recovers base feature values from a one-hot encoded data point.

        Integer-valued numerical features are returned as ``int``.
        """
        row = val.check_row(row, self.n_columns)
        values = {}
        for j, (name, category) in enumerate(self.columns):
            if category is None:
                v = float(row[j])
                values[name] = int(v) if v.is_integer() else v
            elif row[j] == 1:
                values[name] = category
        return values

    def encode(self, values):
        """One-hot encodes a dictionary of base feature values."""
        import numpy as np
        row = np.zeros(self.n_columns, dtype=np.float64)
        for j, (name, category) in enumerate(self.columns):
            if category is None:
                row[j] = float(values[name])
            else:
                row[j] = float(str(values.get(name)) == category)
        return row
