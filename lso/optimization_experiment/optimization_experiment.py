from abc import ABC

from lso.data_manager import data_manager as lso_data_manager
from lso.model_manager import model_manager as lso_model_manager
from lso.optimizer_manager import optimizer_manager as lso_optimizer_manager
from lso.objective_function import objective_function as lso_objective_function


class OptimizationExperimentResult:

    def __init__(
            self,
            data_manager: lso_data_manager.DataManager,
            model_manager: lso_model_manager.ModelManager,
            optimizer_manager: lso_optimizer_manager.OptimizerManager,
            objective_function: lso_objective_function.ObjectiveFunction
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.optimizer_manager = optimizer_manager
        self.objective_function = objective_function


class OptimizationExperiment(ABC):

    def run(
            self,
            data_manager: lso_data_manager.DataManager,
            model_manager: lso_model_manager.ModelManager,
            optimizer: lso_optimizer_manager.OptimizerManager,
            objective_function: lso_objective_function.ObjectiveFunction
    ) -> OptimizationExperimentResult:
        raise NotImplementedError
