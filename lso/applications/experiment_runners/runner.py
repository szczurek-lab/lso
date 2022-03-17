from typing import Dict

from lso.applications.data_managers import factory as lso_dm_factory
from lso.applications.experiments import factory as lso_app_exp_factory
from lso.applications.model_managers import factory as lso_app_mm_factory
from lso.applications.objectives import factory as lso_app_ob_factory
from lso.applications.optimizer_managers import factory as lso_app_om_factory


def run_experiment_from_args(args: Dict):
    experiment = lso_app_exp_factory.OptimizationExperimentFactory.get_from_args(args['optimization_experiment'])
    model_manager = lso_app_mm_factory.ModelManagerFactory.get_from_args(args['model_managers'])
    data_manager = lso_dm_factory.DataManagerFactory.get_from_args(args['data_manager'])
    optimizer_manager = lso_app_om_factory.OptimizerManagerFactory.get_from_args(args['optimization_manager'])
    objective_function = lso_app_ob_factory.ObjectiveFactory.get_from_args(args['objective_function'])
    return experiment.run(
        model_manager=model_manager,
        data_manager=data_manager,
        objective_function=objective_function,
        optimizer_manager=optimizer_manager,
    )
