import os
from functools import singledispatch
from typing import List

import torch

from lso.applications.models.torch import torch_model as lso_tm
from lso.model import model as lso_model
from lso.utils import io as lso_io_utils


TORCH_STATE_DICT_FILE_NAME = 'torch_state_dict.pt'


@singledispatch
def save_instance_params(instance_params: lso_model.InstanceParams, path: str):
    raise NotImplementedError(f'Instance params serialization not implemented for {type(instance_params).__name__}.')


@save_instance_params.register(lso_tm.PytorchInstanceParams)
def save_pytorch_instance_params(instance_params: lso_tm.PytorchInstanceParams, path: str):
    lso_io_utils.create_path(path)
    lso_io_utils.save_type_name_to_path(instance_params, path)
    full_pytorch_state_dict_file_name = os.path.join(path, TORCH_STATE_DICT_FILE_NAME)
    torch.save(instance_params.state_dict, full_pytorch_state_dict_file_name)


def load_instance_params(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    # Switch w.r.t. to different instance params below:
    if type_name == 'PytorchInstanceParams':
        return load_pytorch_instance_params_from_path(path=path)
    raise ValueError(f'Unknown type of instance params: {type_name}')


def load_pytorch_instance_params_from_path(path: str):
    full_pytorch_state_dict_file_name = os.path.join(path, TORCH_STATE_DICT_FILE_NAME)
    state_dict = torch.load(full_pytorch_state_dict_file_name)
    return lso_tm.PytorchInstanceParams(state_dict=state_dict)


def save_instances_params(instances_params: List[lso_model.InstanceParams], path: str):
    lso_io_utils.create_path(path)
    for idx, instance_params in enumerate(instances_params):
        current_instance_params_path = os.path.join(path, str(idx))
        save_instance_params(instances_params[idx], current_instance_params_path)


def load_instances_params(path: str):
    nb_of_instances = len(os.listdir(path))
    instances_params = []

    for idx in range(nb_of_instances):
        current_instance_params_path = os.path.join(path, str(idx))
        instances_params.append(load_instance_params(current_instance_params_path))

    return instances_params
