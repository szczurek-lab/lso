import os
from functools import singledispatch

import torch

from lso.applications.models.torch import torch_instance_params as lso_tip
from lso.model import model as lso_model
from lso.utils import io as lso_io_utils


TORCH_STATE_DICT_FILE_NAME = 'torch_state_dict.pt'


@singledispatch
def save_instance_params(instance_params: lso_model.InstanceParams, path: str):
    raise NotImplementedError(f'Instance params serialization not implemented for {type(instance_params).__name__}.')


def load_instance_params(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    # Switch w.r.t. to different instance params below:
    if type_name == 'PytorchInstanceParams':
        return load_pytorch_instance_params_from_path(path=path)
    raise ValueError(f'Unknown type of instance params: {type_name}')


def load_pytorch_instance_params_from_path(path: str):
    full_pytorch_state_dict_file_name = os.path.join(path, TORCH_STATE_DICT_FILE_NAME)
    state_dict = torch.load(full_pytorch_state_dict_file_name)
    return lso_tip.PytorchInstanceParams(state_dict=state_dict)
