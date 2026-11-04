import json
import os

import pytest

from ctreex import Feature, FeatureSpec, TreeStructure

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def load_case(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


@pytest.fixture
def toy_tree():
    """
    age <= 30 -> (job_a <= 0.5 -> 'no' (leaf 2), else 'yes' (leaf 3));
    age > 30  -> (age <= 50 -> 'yes' (leaf 5), else 'no' (leaf 6)).
    """
    return TreeStructure(
        children_left=[1, 2, -1, -1, 5, -1, -1],
        children_right=[4, 3, -1, -1, 6, -1, -1],
        feature=[0, 1, -2, -2, 0, -2, -2],
        threshold=[30., 0.5, -2, -2, 50., -2, -2],
        value=[[6, 6], [3, 3], [3, 0], [0, 3], [3, 3], [0, 3], [3, 0]],
        classes=['no', 'yes'], n_features=4,
        feature_names=['age', 'job_a', 'job_b', 'job_c'])


@pytest.fixture
def toy_spec():
    return FeatureSpec.from_column_names(
        ['age', 'job_a', 'job_b', 'job_c'],
        features=[Feature('job', 'categorical', display_name='job type',
                          value_names={'a': 'artist', 'b': 'baker',
                                       'c': 'clerk'})])
