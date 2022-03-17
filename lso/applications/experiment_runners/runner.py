from typing import Dict

from lso.applications.data_managers import factory as lso_dm_factory
from lso.applications.experiments import factory as lso_app_exp_factory
from lso.applications.model_managers import factory as lso_app_mm_factory
from lso.applications.objectives import factory as lso_app_ob_factory
from lso.applications.optimizer_managers import factory as lso_app_om_factory
from lso.optimization_experiment import optimization_experiment as lso_oe


DATA_MANAGER_ARG_NAME = 'data_manager'
MODEL_MANAGER_ARG_NAME = 'model_manager'
OBJECTIVE_FUNCTION_ARG_NAME = 'objective_function'
OPTIMIZATION_EXPERIMENT_ARG_NAME = 'optimization_experiment'
OPTIMIZER_MANAGER_ARG_NAME = 'optimization_manager'


def run_experiment_from_args(args: Dict) -> lso_oe.OptimizationExperimentResult:
    experiment = lso_app_exp_factory.OptimizationExperimentFactory.get_from_args(
        args=args[OPTIMIZATION_EXPERIMENT_ARG_NAME],
    )
    model_manager = lso_app_mm_factory.ModelManagerFactory.get_from_args(
        args=args[MODEL_MANAGER_ARG_NAME],
    )
    data_manager = lso_dm_factory.DataManagerFactory.get_from_args(
        args=args[DATA_MANAGER_ARG_NAME],
    )
    optimizer_manager = lso_app_om_factory.OptimizerManagerFactory.get_from_args(
        args=args[OPTIMIZER_MANAGER_ARG_NAME],
    )
    objective_function = lso_app_ob_factory.ObjectiveFactory.get_from_args(args[OBJECTIVE_FUNCTION_ARG_NAME])
    return experiment.run(
        model_manager=model_manager,
        data_manager=data_manager,
        objective_function=objective_function,
        optimizer_manager=optimizer_manager,
    )
