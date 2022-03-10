from abc import ABC

from lso.data import data as lso_data


class ObjectiveFunction(ABC):

    def evaluate(self, data: lso_data.Data) -> lso_data.Data:
        raise NotImplementedError
