from functools import singledispatch
import os

from lso.applications.data_managers import serialization as lso_app_dm_serialization
from lso.applications.experiments import base_experiments as lso_app_exp_be
from lso.applications.model_managers import serialization as lso_app_mm_serialization
from lso.applications.objectives import serialization as lso_app_obj_serialization
from lso.applications.optimizer_managers import serialization as lso_app_om_serialization
from lso.optimization_experiment import optimization_experiment as lso_oe
from lso.utils import io as lso_io_utils


DATA_MANAGER_SUBPATH = 'data_manager'
MODEL_MANAGER_SUBPATH = 'model_manager'
OPTIMIZATION_MANAGER_SUBPATH = 'optimization_manager'
OBJECTIVE_SUBPATH = 'objective'


@singledispatch
def save_experiment_result(experiment_result: lso_oe.OptimizationExperimentResult, path: str):
    raise ValueError(f'ExperimentResult serialization not defined for {experiment_result}.')


@save_experiment_result.register(lso_app_exp_be.BaseOptimizationExperimentResult)
def save_base_experiment_result(experiment_result: lso_app_exp_be.BaseOptimizationExperimentResult, path: str):
    lso_io_utils.create_path(path=path)
    lso_io_utils.save_type_name_to_path(obj=experiment_result, path=path)
    lso_io_utils.save_config_dict_to_path(experiment_result.get_config_dict(), path=path)

    data_manager_path = os.path.join(path, DATA_MANAGER_SUBPATH)
    model_manager_path = os.path.join(path, MODEL_MANAGER_SUBPATH)
    optimization_manager_path = os.path.join(path, OPTIMIZATION_MANAGER_SUBPATH)
    objective_function_path = os.path.join(path, OBJECTIVE_SUBPATH)

    lso_io_utils.create_path(data_manager_path)
    lso_app_dm_serialization.save_data_manager(experiment_result.data_manager, path=data_manager_path)
    lso_io_utils.create_path(model_manager_path)
    lso_app_mm_serialization.save_model_manager(experiment_result.model_manager, path=model_manager_path)
    lso_io_utils.create_path(optimization_manager_path)
    lso_app_om_serialization.save_optimizer_manager(experiment_result.optimizer_manager, path=optimization_manager_path)
    lso_io_utils.create_path(objective_function_path)
    lso_app_obj_serialization.save_objective_function(
        experiment_result.objective_function, path=objective_function_path,
    )


def load_optimization_experiment(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    if type_name == 'BaseOptimizationExperimentResult':
        return load_base_optimization_experiment_result(path=path)
    raise ValueError(f'Unknown deserialization procedure for a {type_name}.')


def load_base_optimization_experiment_result(path: str):
    config_dict = lso_io_utils.load_config_dict_from_path(path)
    data_manager_path = os.path.join(path, DATA_MANAGER_SUBPATH)
    model_manager_path = os.path.join(path, MODEL_MANAGER_SUBPATH)
    optimization_manager_path = os.path.join(path, OPTIMIZATION_MANAGER_SUBPATH)
    objective_function_path = os.path.join(path, OBJECTIVE_SUBPATH)

    return lso_app_exp_be.BaseOptimizationExperimentResult(
        data_manager=lso_app_dm_serialization.load_data_manager(data_manager_path),
        model_manager=lso_app_mm_serialization.load_model_manager(model_manager_path),
        objective_function=lso_app_obj_serialization.load_objective_function(objective_function_path),
        optimizer_manager=lso_app_om_serialization.load_optimizer_manager(optimization_manager_path),
        **config_dict,
    )
