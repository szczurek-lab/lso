import numpy as np

from lso.data import data as lso_data


class DummyNumpyData(lso_data.NumpyData):
    DUMMY_X_SHAPE = 10
    DUMMY_OBJECTIVE_SHAPE = 1
    DUMMY_FEATURES_SHAPE = 15


def test__if_adds_the_same_type_with_not_nones():
    left_size = 15
    right_size = 30
    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    result = left + right
    assert np.allclose(result.x[:left_size], left.x)
    assert np.allclose(result.x[left_size:], right.x)
    assert np.allclose(result.objective[:left_size], left.objective)
    assert np.allclose(result.objective[left_size:], right.objective)
    assert np.allclose(result.features[:left_size], left.features)
    assert np.allclose(result.features[left_size:], right.features)


def test__if_adds_the_same_type_with_single_objective_none():
    left_size = 15
    right_size = 30
    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=None,
        features=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    result = left + right
    assert result.objective is None
    result = right + left
    assert result.objective is None


def test__if_adds_the_same_type_with_both_objective_none():
    left_size = 15
    right_size = 30
    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=None,
        features=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=None,
        features=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    result = left + right
    assert result.objective is None
    result = right + left
    assert result.objective is None


def test__if_adds_the_same_type_with_single_features_none():
    left_size = 15
    right_size = 30
    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=None,
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_FEATURES_SHAPE))
    )
    result = left + right
    assert result.features is None
    result = right + left
    assert result.features is None


def test__if_adds_the_same_type_with_both_features_none():
    left_size = 15
    right_size = 30
    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(left_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=None,
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_X_SHAPE)),
        objective=np.random.normal(size=(right_size, DummyNumpyData.DUMMY_OBJECTIVE_SHAPE)),
        features=None,
    )
    result = left + right
    assert result.features is None
    result = right + left
    assert result.features is None
