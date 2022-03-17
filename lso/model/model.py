from typing import Dict
from typing import Optional

from lso.data import data as lso_data
from lso.data import numpy_data as lso_np_data


class InstanceParams:
    pass


class ModelInstance:

    def get_params(self):
        raise NotImplementedError

    def encode(self, data: lso_np_data.NumpyData) -> lso_data.Latent:
        raise NotImplementedError

    def decode(self, latent: lso_data.Latent) -> lso_np_data.NumpyData:
        raise NotImplementedError


class Model:

    def get_instance(self, instance_params: Optional[InstanceParams] = None) -> ModelInstance:
        raise NotImplementedError

    def get_config_dict(self) -> Dict:
        raise NotImplementedError

    @classmethod
    def from_config_dict(cls, config_dict: Dict) -> "Model":
        raise NotImplementedError
