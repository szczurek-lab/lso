from typing import Dict

import tqdm

from lso.data_manager import data_manager as lso_data_manager
from lso.optimization_experiment import optimization_experiment as lso_oe
from lso.model_manager import model_manager as lso_model_manager
from lso.objective_function import objective_function as lso_objective_function
from lso.optimizer_manager import optimizer_manager as lso_optimizer_manager


class BaseOptimizationExperimentResult(lso_oe.OptimizationExperimentResult):

    def __init__(
            self,
            model_manager: lso_model_manager.ModelManager,
            data_manager: lso_data_manager.DataManager,
            optimizer_manager: lso_optimizer_manager.OptimizerManager,
            nb_of_epochs: int,
            nb_of_candidates_per_epoch: int
    ):
        super().__init__(
            model_manager=model_manager,
            data_manager=data_manager,
            optimizer_manager=optimizer_manager,
        )
        self.nb_of_epochs = nb_of_epochs
        self.nb_of_candidates_per_epoch = nb_of_candidates_per_epoch


class BaseOptimizationExperiment(lso_oe.OptimizationExperiment):

    def __init__(self, nb_of_epochs: int, nb_of_candidates_per_epoch: int):
        self.nb_of_epochs = nb_of_epochs
        self.nb_of_candidates_per_epoch = nb_of_candidates_per_epoch

    def run(
            self,
            data_manager: lso_data_manager.DataManager,
            model_manager: lso_model_manager.ModelManager,
            optimizer_manager: lso_optimizer_manager.OptimizerManager,
            objective_function: lso_objective_function.ObjectiveFunction,
    ):
        for epoch_nb in tqdm.tqdm(range(self.nb_of_epochs)):
            model_manager.train(data_manager, epoch_nb)
            candidates = optimizer_manager.get_candidates(
                nb_of_candidates=self.nb_of_candidates_per_epoch,
                model_manager=model_manager,
                data_manager=data_manager,
                epoch_nb=epoch_nb,
            )
            data_manager.append(objective_function.evaluate(candidates))

        return BaseOptimizationExperimentResult(
            data_manager=data_manager,
            model_manager=model_manager,
            optimizer_manager=optimizer_manager,
            nb_of_epochs=self.nb_of_epochs,
            nb_of_candidates_per_epoch=self.nb_of_candidates_per_epoch
        )

    def get_config_dict(self) -> Dict:
        return {'nb_of_epochs': self.nb_of_epochs, 'nb_of_candidates_per_epochs': self.nb_of_candidates_per_epoch}
