from abc import ABC

from lso.data import data as lso_data
from lso.data import numpy_data as lso_np_data


class Vectorizer(ABC):

    def encode(self, data: lso_data.Data) -> lso_np_data.NumpyData:
        raise NotImplementedError

    def decode(self, data: lso_np_data.NumpyData) -> lso_data.Data:
        raise NotImplementedError

    def get_config_dict(self):
        raise NotImplementedError


class IdentityVectorizer(Vectorizer):

    def encode(self, data: lso_data.Data) -> lso_np_data.NumpyData:
        if isinstance(data, lso_np_data.NumpyData):
            return data
        raise ValueError(f'Data provided to a IdentityVectorizer should be a NumpyData')

    def decode(self, data: lso_np_data.NumpyData) -> lso_data.Data:
        return data

    def get_config_dict(self):
        return {}
