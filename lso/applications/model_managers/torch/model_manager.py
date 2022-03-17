from typing import List, Dict
from typing import Optional

import pytorch_lightning as pl
from torch.utils import data as torch_data

from lso.applications.models.torch import torch_model as lso_tm
from lso.data import vectorizer as lso_vectorizer
from lso.data_manager import data_manager as lso_dm
from lso.model_manager import model_manager as lso_model_manager


class BasicPLSingleModelModelManager(lso_model_manager.SingleModelModelManager):

    def __init__(
            self,
            model: lso_tm.PytorchModel,
            vectorizer: lso_vectorizer.Vectorizer,
            batch_size: int = 32,
            nb_of_initial_epochs: int = 50,
            nb_of_steps_per_epoch: int = 1,
            instance_params: Optional[List[lso_tm.PytorchInstanceParams]] = None,
    ):
        super().__init__(model=model, instance_params=instance_params, vectorizer=vectorizer)
        self.batch_size = batch_size
        self.nb_of_step_epochs = nb_of_steps_per_epoch
        self.nb_of_initial_epochs = nb_of_initial_epochs

    def train(self, data_manager: lso_dm.DataManager, epoch_nb: int):
        np_data = self.vectorizer.encode(data_manager.get_all_data())
        train = torch_data.DataLoader(np_data.x, batch_size=self.batch_size, shuffle=True)
        model_instance = self.model.get_instance() if epoch_nb == 0 else \
            self.model.get_instance(self.instance_params[epoch_nb - 1])
        max_epochs = self.nb_of_initial_epochs if epoch_nb == 0 else self.nb_of_step_epochs
        trainer = pl.Trainer(
            gpus=0,
            max_epochs=max_epochs,
            progress_bar_refresh_rate=20,
        )
        trainer.fit(model_instance, train)
        self.instance_params.append(model_instance.get_params())

    def get_config_dict(self) -> Dict:
        return {
            'batch_size': self.batch_size,
            'nb_of_step_epochs': self.nb_of_step_epochs,
            'nb_of_initial_epochs': self.nb_of_initial_epochs,
            'vectorizer': self.vectorizer.get_config_dict(),
        }
