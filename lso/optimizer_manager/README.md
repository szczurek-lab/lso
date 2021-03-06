# Optimizer Manager

In this directory we will implement implementations of `OptimizerManager`. The base interface of
`OptimizerManager` aims at modelling how optimization algorithm looks for new candidates (which are new `Data` points)
during the optimization procedure.

## `OptimizerManager`
- `get_candidates(self, model_manager: lso_model_manager.ModelManager, data_manager: lso_data_manager.DataManager, epoch_nb: int, nb_of_candidates: int) -> lso_data.Data`:
get `nb_of_candidates` candidates for epoch `epoch_nb` given the data provided so far in form of a `DataManager` and models provided so far in form of
a `ModelManager`. 
