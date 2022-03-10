from functools import singledispatch

from lso.model import model as lso_model


@singledispatch
def save_instance_params(instance_params: lso_model.InstanceParams, path: str):
    raise NotImplementedError(f'Instance params serialization not implemented for {type(instance_params).__name__}.')
