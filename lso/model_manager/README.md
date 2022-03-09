# Model Managers

In this folder we provide implementation for `ModelManager` classes. A base interface of a `ModelManager`
aims at providing appropriate object for storing model trajectories collected during the optimization experiments.

## `ModelManager`

- `train(self, data_manager: DataManager, epoch_nb: int)`: extend the model trajectory by training a new model 
on a data provided by a `data_manager:DataManager` object.
- `encode(self, data: Data, epoch_nb: int) -> Latent`: get the encoding of data from a model from an 
`epoch_nb` epoch.
- `decode(self, data: Latent, epoch_nb: int) -> Data`: decode the latent back to the data space using
a model from an `epoch_nb` epoch.
