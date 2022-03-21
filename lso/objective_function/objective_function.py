from abc import ABC
from typing import Dict

from lso.data import data as lso_data


class ObjectiveFunction(ABC):

    def evaluate(self, data: lso_data.Data) -> lso_data.Data:
        raise NotImplementedError

    def get_config_dict(self) -> Dict:
        raise NotImplementedError
