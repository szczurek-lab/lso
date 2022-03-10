from functools import singledispatch

from lso.model import model as lso_model
from lso.utils import io as lso_io_utils

# Insert new model types here:
MODEL_TYPES = []
MODEL_NAME_TO_TYPE = {type_.__name__: type_ for type_ in MODEL_TYPES}


@singledispatch
def save_model(model: lso_model.Model, path: str):
    raise NotImplementedError(f'Model serialization not implemented for {type(model).__name__}.')


def load_model(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    config_dict = lso_io_utils.load_config_dict_from_path(path=path)
    return MODEL_NAME_TO_TYPE[type_name].from_config_dict(config_dict)
