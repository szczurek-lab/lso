from functools import singledispatch
import os

from lso.applications.data import vectorizer_serialization
from lso.applications.model_managers.torch import model_manager as lso_app_mm_t
from lso.applications.models import instance_params_serialization
from lso.applications.models import model_serialization
from lso.model_manager import model_manager as lso_mm
import lso.utils.io as lso_io_utils


MODEL_SUBPATH = 'model'
INSTANCES_PARAMS_SUBPATH = 'instance_params'
VECTORIZER_SUBPATH = 'vectorizer'


@singledispatch
def save_model_manager(model_manager: lso_mm.ModelManager, path: str):
    raise ValueError(f'ModelManager serialization not defined for {model_manager}.')


@save_model_manager.register(lso_app_mm_t.BasicPLSingleModelModelManager)
def save_single_instance_pytorch_model_manager(model_manager: lso_app_mm_t.BasicPLSingleModelModelManager, path: str):
    lso_io_utils.create_path(path=path)
    lso_io_utils.save_type_name_to_path(obj=model_manager, path=path)
    lso_io_utils.save_config_dict_to_path(config_dict=model_manager.get_config_dict(), path=path)

    model_subpath = os.path.join(path, MODEL_SUBPATH)
    instances_params_subpath = os.path.join(path, INSTANCES_PARAMS_SUBPATH)
    vectorizer_subpath = os.path.join(path, VECTORIZER_SUBPATH)
    lso_io_utils.create_path(model_subpath)
    lso_io_utils.create_path(instances_params_subpath)
    lso_io_utils.create_path(vectorizer_subpath)

    model_serialization.save_model(model_manager.model, path=model_subpath)
    instance_params_serialization.save_instances_params(model_manager.instance_params, path=instances_params_subpath)
    vectorizer_serialization.save_vectorizer(model_manager.vectorizer, path=vectorizer_subpath)


def load_model_manager(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    if type_name == 'BasicPLSingleModelModelManager':
        return load_single_instance_pytorch_model_manager(path=path)
    raise ValueError(f'Unknown deserialization procedure for a {type_name}.')


def load_single_instance_pytorch_model_manager(path: str):
    model_subpath = os.path.join(path, MODEL_SUBPATH)
    instances_params_subpath = os.path.join(path, INSTANCES_PARAMS_SUBPATH)
    vectorizer_subpath = os.path.join(path, VECTORIZER_SUBPATH)

    model = model_serialization.load_model(model_subpath)
    instances_params = instance_params_serialization.load_instances_params(instances_params_subpath)
    vectorizer = vectorizer_serialization.load_vectorizer(vectorizer_subpath)
    config_dict = lso_io_utils.load_config_dict_from_path(path=path)
    return lso_app_mm_t.BasicPLSingleModelModelManager(
        model=model,
        instance_params=instances_params,
        vectorizer=vectorizer,
        **config_dict,
    )
