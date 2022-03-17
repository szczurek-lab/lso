from abc import ABC
from typing import Dict
from typing import List
from typing import Optional

from lso.data import data as lso_data
from lso.data import vectorizer as lso_vectorizer
from lso.data_manager import data_manager as lso_dm
from lso.model import model as lso_model


class ModelManager(ABC):

    def __init__(self, vectorizer: lso_vectorizer.Vectorizer):
        self.vectorizer = vectorizer

    def train(self, data_manager: lso_dm.DataManager, epoch_nb: int):
        raise NotImplementedError

    def encode(self, data: lso_data.Data, epoch_nb: int) -> lso_data.Latent:
        raise NotImplementedError

    def decode(self, data: lso_data.Latent, epoch_nb: int) -> lso_data.Data:
        raise NotImplementedError

    def get_config_dict(self) -> Dict:
        raise NotImplementedError


class SingleModelModelManager(ModelManager, ABC):

    def __init__(
            self,
            model: lso_model.Model,
            vectorizer: lso_vectorizer.Vectorizer,
            instance_params: Optional[List] = None,
    ):
        super().__init__(vectorizer=vectorizer)
        self.model = model
        self.instance_params = instance_params if instance_params is not None else []

    def train(self, data_manager: lso_dm.DataManager, epoch_nb: int):
        raise NotImplementedError

    def encode(self, data: lso_data.Data, epoch_nb: int) -> lso_data.Latent:
        model_instance = self.model.get_instance(self.instance_params[epoch_nb])
        vectorized_data = self.vectorizer.encode(data)
        return model_instance.encode(data=vectorized_data)

    def decode(self, latent: lso_data.Latent, epoch_nb) -> lso_data.Data:
        model_instance = self.model.get_instance(self.instance_params[epoch_nb])
        vectorized_data = model_instance.decode(latent=latent)
        return self.vectorizer.decode(vectorized_data)
