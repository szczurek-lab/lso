import numpy as np
import pytest

from lso.data import numpy_data as lso_np_data


def test__numpy_data__add_without_nulls():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    result = left + right
    assert np.allclose(result.x[:left_size], left.x)
    assert np.allclose(result.x[-right_size:], right.x)
    assert np.allclose(result.features[:left_size], left.features)
    assert np.allclose(result.features[-right_size:], right.features)
    assert np.allclose(result.objective[:left_size], left.objective)
    assert np.allclose(result.objective[-right_size:], right.objective)

    result = right + left
    assert np.allclose(result.x[:right_size], right.x)
    assert np.allclose(result.x[-left_size:], left.x)
    assert np.allclose(result.features[:right_size], right.features)
    assert np.allclose(result.features[-left_size:], left.features)
    assert np.allclose(result.objective[:right_size], right.objective)
    assert np.allclose(result.objective[-left_size:], left.objective)


def test__numpy_data__add_with_objective_nulls():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=None,
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    result = left + right
    assert result.objective is None
    result = right + left
    assert result.objective is None


def test__numpy_data__add_with_features_nulls():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=None,
    )
    result = left + right
    assert result.features is None
    result = right + left
    assert result.features is None


def test__numpy_latent__add_without_nulls():
    left_size = 10
    right_size = 10
    z_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, z_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    result = left + right
    assert np.allclose(result.z[:left_size], left.z)
    assert np.allclose(result.z[-right_size:], right.z)
    assert np.allclose(result.features[:left_size], left.features)
    assert np.allclose(result.features[-right_size:], right.features)
    assert np.allclose(result.objective[:left_size], left.objective)
    assert np.allclose(result.objective[-right_size:], right.objective)

    result = right + left
    assert np.allclose(result.z[:right_size], right.z)
    assert np.allclose(result.z[-left_size:], left.z)
    assert np.allclose(result.features[:right_size], right.features)
    assert np.allclose(result.features[-left_size:], left.features)
    assert np.allclose(result.objective[:right_size], right.objective)
    assert np.allclose(result.objective[-left_size:], left.objective)


def test__numpy_latent__add_with_objective_nulls():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, x_shape)),
        objective=None,
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    result = left + right
    assert result.objective is None
    result = right + left
    assert result.objective is None


def test__numpy_latent__add_with_features_nulls():
    left_size = 10
    right_size = 10
    z_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, z_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=None,
    )
    result = left + right
    assert result.features is None
    result = right + left
    assert result.features is None


def test__numpy_latent_and_numpy_data__not_implemented():
    left_size = 10
    right_size = 10
    z_shape = 15
    x_shape = 15
    features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=None,
    )
    with pytest.raises(TypeError):
        left + right
        right + left


def test__numpy_data__raises_when_x_shapes_wrong():
    left_size = 10
    right_size = 10
    left_x_shape = 15
    right_x_shape = 11
    features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, left_x_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, right_x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_data__raises_when_features_shapes_wrong():
    left_size = 10
    right_size = 10
    x_shape = 15
    left_features_shape = 11
    right_features_shape = 20
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, left_features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, right_features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_data__raises_when_objective_shapes_wrong():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 11
    left_objective_shape = 13
    right_objective_shape = 11
    left = lso_np_data.NumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, left_objective_shape)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, right_objective_shape)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_latent__raises_when_x_shapes_wrong():
    left_size = 10
    right_size = 10
    left_z_shape = 15
    right_z_shape = 11
    features_shape = 20
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, left_z_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, right_z_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_latent__raises_when_features_shapes_wrong():
    left_size = 10
    right_size = 10
    z_shape = 15
    left_features_shape = 11
    right_features_shape = 20
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, z_shape)),
        objective=np.random.normal(size=(left_size, 1)),
        features=np.random.normal(size=(left_size, left_features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, 1)),
        features=np.random.normal(size=(right_size, right_features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_latent__raises_when_objective_shapes_wrong():
    left_size = 10
    right_size = 10
    z_shape = 15
    features_shape = 11
    left_objective_shape = 13
    right_objective_shape = 11
    left = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(left_size, z_shape)),
        objective=np.random.normal(size=(left_size, left_objective_shape)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, right_objective_shape)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    with pytest.raises(ValueError):
        left + right
        right + left


def test__numpy_latent__subclass_proper_when_adding():
    left_size = 10
    right_size = 10
    z_shape = 15
    features_shape = 11
    objective_shape = 11

    class DummyNumpyLatent(lso_np_data.NumpyLatent):
        pass

    left = DummyNumpyLatent(
        z=np.random.normal(size=(left_size, z_shape)),
        objective=np.random.normal(size=(left_size, objective_shape)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = lso_np_data.NumpyLatent(
        z=np.random.normal(size=(right_size, z_shape)),
        objective=np.random.normal(size=(right_size, objective_shape)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    assert isinstance(left + right, DummyNumpyLatent)


def test__numpy_data__subclass_proper_when_adding():
    left_size = 10
    right_size = 10
    x_shape = 15
    features_shape = 11
    objective_shape = 11

    class DummyNumpyData(lso_np_data.NumpyData):
        pass

    left = DummyNumpyData(
        x=np.random.normal(size=(left_size, x_shape)),
        objective=np.random.normal(size=(left_size, objective_shape)),
        features=np.random.normal(size=(left_size, features_shape)),
    )
    right = DummyNumpyData(
        x=np.random.normal(size=(right_size, x_shape)),
        objective=np.random.normal(size=(right_size, objective_shape)),
        features=np.random.normal(size=(right_size, features_shape)),
    )
    assert isinstance(left + right, DummyNumpyData)
