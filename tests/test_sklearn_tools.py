import numpy as np
import pytest

sklearn_tree = pytest.importorskip('sklearn.tree')

from ctreex import sklearn_tools as st  # noqa: E402


@pytest.fixture
def clf():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 3)
    y = np.array(['n', 'n', 'n', 'y'] * 3)
    return sklearn_tree.DecisionTreeClassifier(random_state=0).fit(X, y)


def test_text_and_dot(clf):
    text = st.decision_tree_text(clf, ['a', 'b'],
                                 class_map={'n': 'no', 'y': 'yes'})
    assert 'class: yes' in text and 'class: no' in text
    dot = st.decision_tree_dot(clf, ['a', 'b'], class_names=['no', 'yes'])
    assert dot.startswith('digraph')


def test_validation(clf):
    with pytest.raises(ValueError):
        st.decision_tree_text(clf, ['a'])
    with pytest.raises(TypeError):
        st.decision_tree_text(clf, 'a')
    with pytest.raises(ValueError):
        st.decision_tree_dot(clf, class_map={'x': 'y'})
    with pytest.raises(TypeError):
        st.decision_tree_feature_importance(object())


def test_root2leaf(clf):
    leaf = int(clf.apply(np.array([[1, 1]]))[0])
    path = st.decision_tree_root2leaf(clf, leaf)
    assert path[0] == 0 and path[-1] == leaf
    assert st.decision_tree_feature_importance(clf).shape == (2, )
