from functools import singledispatch

from lso.applications.optimizer_managers import sample_around_top_k as lso_app_om_satk
from lso.optimizer_manager import optimizer_manager as lso_om
from lso.utils import io as lso_io


IMPLEMENTED_OPTIMIZER_MANAGERS_TYPES = [
    lso_app_om_satk.SampleAroundTopKOptimizerManager
]

TYPE_NAME_TO_IMPLEMENTED_OPTIMIZER_MANAGERS_TYPES = {
    type_.__name__: type_ for type_ in IMPLEMENTED_OPTIMIZER_MANAGERS_TYPES
}


def load_optimizer_manager(path: str):
    type_ = lso_io.load_type_name_from_path(path=path)
    if type_ not in TYPE_NAME_TO_IMPLEMENTED_OPTIMIZER_MANAGERS_TYPES:
        raise ValueError(f'Unknown OptimizerManager type {type_}.')
    config_dict = lso_io.load_config_dict_from_path(path=path)
    return TYPE_NAME_TO_IMPLEMENTED_OPTIMIZER_MANAGERS_TYPES[type_](**config_dict)


@singledispatch
def save_optimizer_manager(optimizer_manager: lso_om.OptimizerManager, path: str):
    raise NotImplementedError(f'There is no default way for serialization of {optimizer_manager}.')


@save_optimizer_manager.register(lso_app_om_satk.SampleAroundTopKOptimizerManager)
def save_satk(optimizer_manager: lso_app_om_satk.SampleAroundTopKOptimizerManager, path: str):
    lso_io.create_path(path=path)
    lso_io.save_type_name_to_path(obj=optimizer_manager, path=path)
    lso_io.save_config_dict_to_path(config_dict=optimizer_manager.get_config_dict(), path=path)
