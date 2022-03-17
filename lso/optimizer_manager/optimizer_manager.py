from abc import ABC

from lso.data import data as lso_data
from lso.data_manager import data_manager as lso_data_manager
from lso.model_manager import model_manager as lso_model_manager


class OptimizerManager(ABC):

    def get_candidates(
        self,
        model_manager: lso_model_manager.ModelManager,
        data_manager: lso_data_manager.DataManager,
        epoch_nb: int,
        nb_of_candidates: int
    ) -> lso_data.Data:
        raise NotImplementedError

    def get_config_dict(self):
        raise NotImplementedError
