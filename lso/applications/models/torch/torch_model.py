from abc import ABC
from typing import Dict
from typing import Optional

import pytorch_lightning as pl

from lso.model import model as lso_model


class PytorchInstanceParams(lso_model.InstanceParams):

    def __init__(self, state_dict: Dict):
        self.state_dict = state_dict


class PytorchModelInstance(pl.LightningModule, lso_model.ModelInstance, ABC):
    pass


class PytorchModel(lso_model.Model, ABC):

    def get_instance(self, instance_params: Optional[PytorchInstanceParams] = None) -> PytorchModelInstance:
        raise NotImplementedError
