from abc import ABC
from typing import List
from typing import Optional

from lso.data import data as lso_data


class DataManager(ABC):

    def append(self, data: lso_data.Data) -> None:
        raise NotImplementedError

    def get_initial_data(self) -> lso_data.Data:
        raise NotImplementedError

    def get_nb_of_epochs(self) -> int:
        raise NotImplementedError

    def get_data_from_epoch(self, epoch_nb: int) -> lso_data.Data:
        raise NotImplementedError

    def get_all_data(self) -> lso_data.Data:
        return sum(
            [self.get_data_from_epoch(epoch_nb) for epoch_nb in range(self.get_nb_of_epochs())],
            self.get_initial_data(),
        )

    def get_data_from_epochs(self) -> lso_data.Data:
        return sum(
            [self.get_data_from_epoch(epoch_nb) for epoch_nb in range(1, self.get_nb_of_epochs())],
            self.get_data_from_epoch(0),
        )


class BaseDataManager(DataManager):

    def __init__(self, initial_data: lso_data.Data, datas: Optional[List[lso_data.Data]] = None):
        self.initial_data = initial_data
        self.datas = datas if datas is not None else []

    def append(self, data: lso_data.Data):
        self.datas.append(data)

    def get_initial_data(self) -> lso_data.Data:
        return self.initial_data

    def get_nb_of_epochs(self) -> int:
        return len(self.datas)

    def get_data_from_epoch(self, epoch_nb: int) -> lso_data.Data:
        return self.datas[epoch_nb]
