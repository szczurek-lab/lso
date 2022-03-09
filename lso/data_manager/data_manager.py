from lso.data import data


class DataManager(ABC):

    def append(self, data: data.Data) -> None:
        raise NotImplementedError

    def get_initial_data(self) -> data.Data:
        return self.get_initial_data()

    def get_nb_of_epochs(self) -> int:
        raise NotImplementedError

    def get_data_from_epoch(self, epoch_nb: int) -> Data:
        raise NotImplementedError

    def get_all_data(self) -> Data:
        return sum(
            [self.get_data_from_epoch(epoch_nb) for epoch_nb in range(self.get_nb_of_epochs())],
            self.get_initial_data(),
        )

    def get_data_from_epochs(self) -> Data:
        return sum(
            [self.get_data_from_epoch(epoch_nb) for epoch_nb in range(1, self.get_nb_of_epochs())],
            self.get_data_from_epoch(0),
        )
