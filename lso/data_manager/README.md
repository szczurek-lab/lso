# DATA MANAGERS

In this directory we will implement `DataManager` classes. The base interface of  `DataManager`
aims at modeling the optimization trajectory w.r.t. to acquired data points. It has the following interface.

- `append(self, data: Data)`: add a data from new epoch to a data manager.
- `get_initial_data(self)`: get initial data provided to the optimization experiment. 
- `get_nb_of_epochs(self) -> int`: get number of experiment epochs (number of `append` method calls).
- `get_data_from_epoch(self, epoch_nb: int) -> Data`: get data from `epoch_nb` epoch.
- `get_all_data(self) -> Data`: get initial data and data from all epochs.
- `def get_data_from_epochs(self) -> Data`: get data from all optimization epochs.