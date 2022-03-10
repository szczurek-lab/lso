from functools import singledispatch

from lso.model import model as lso_model


@singledispatch
def save_model(model: lso_model.Model, path: str):
    raise NotImplementedError(f'Model serialization not implemented for {type(instance_params).__name__}.')
