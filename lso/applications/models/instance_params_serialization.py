from functools import singledispatch

from lso.model import model as lso_model
from lso.utils import io as lso_io_utils


@singledispatch
def save_instance_params(instance_params: lso_model.InstanceParams, path: str):
    raise NotImplementedError(f'Instance params serialization not implemented for {type(instance_params).__name__}.')


def load_instance_params(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    # Switch w.r.t. to different instance params below:
    raise ValueError(f'Unknown type of instance params: {type_name}')
