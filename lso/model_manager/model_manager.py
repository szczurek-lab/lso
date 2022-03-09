from abc import ABC

from lso.data import data as lso_data
from lso.data_manager import data_manager as lso_dm


class ModelManager(ABC):

    def train(self, data_manager: lso_dm.DataManager, epoch_nb: int):
        raise NotImplementedError

    def encode(self, data: lso_data.Data, epoch_nb: int) -> lso_data.Latent:
        raise NotImplementedError

    def decode(self, data: lso_data.Latent, epoch_nb: int) -> lso_data.Data:
        raise NotImplementedError
