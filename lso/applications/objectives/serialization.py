from functools import singledispatch

from lso.applications.objectives import mnist as lso_app_obj_mnist
from lso.objective_function import objective_function as lso_of
from lso.utils import io as lso_io


IMPLEMENTED_OBJECTIVE_FUNCTION_TYPES = [
    lso_app_obj_mnist.MNISTSumObjectiveFunction,
]

TYPE_NAME_TO_IMPLEMENTED_OBJECTIVE_FUNCTION_TYPES = {
    type_.__name__: type_ for type_ in IMPLEMENTED_OBJECTIVE_FUNCTION_TYPES
}


def load_objective_function(path: str):
    type_ = lso_io.load_type_name_from_path(path=path)
    if type_ not in TYPE_NAME_TO_IMPLEMENTED_OBJECTIVE_FUNCTION_TYPES:
        raise ValueError(f'Unknown ObjectiveFunction type {type_}.')
    config_dict = lso_io.load_config_dict_from_path(path=path)
    return TYPE_NAME_TO_IMPLEMENTED_OBJECTIVE_FUNCTION_TYPES[type_](**config_dict)


@singledispatch
def save_objective_function(objective_function: lso_of.ObjectiveFunction, path: str):
    raise NotImplementedError(f'There is no default way for serialization of {objective_function}.')


@save_objective_function.register(lso_app_obj_mnist.MNISTSumObjectiveFunction)
def save_msof(optimizer_manager: lso_app_obj_mnist.MNISTSumObjectiveFunction, path: str):
    lso_io.create_path(path=path)
    lso_io.save_type_name_to_path(obj=optimizer_manager, path=path)
    lso_io.save_config_dict_to_path(config_dict=optimizer_manager.get_config_dict(), path=path)
