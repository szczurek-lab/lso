from functools import singledispatch
from typing import List

import numpy as np
import os

from lso.applications.data import mnist as lso_app_data_mnist
from lso.data import data as lso_data
from lso.data import numpy_data as lso_np_data
from lso.utils import io as lso_io_utils


# Insert new model types here:
DATA_TYPES = [
    lso_app_data_mnist.MNISTNumpyData,
]
DATA_NAME_TO_TYPE = {type_.__name__: type_ for type_ in DATA_TYPES}

X_FILE_NAME = 'x'
FEATURES_FILE_NAME = 'features'
OBJECTIVE_FILE_NAME = 'objective'

LOAD_X_FILE_NAME_NP = X_FILE_NAME + '.npy'
LOAD_FEATURES_FILE_NAME_NP = FEATURES_FILE_NAME + '.npy'
LOAD_OBJECTIVE_FILE_NAME_NP = OBJECTIVE_FILE_NAME + '.npy'


@singledispatch
def save_data(data: lso_data.Data, path: str):
    if isinstance(data, lso_np_data.NumpyData):
        Warning(f'Default NumpyData serialization run for {type(data).__name__}.')
        lso_io_utils.create_path(path=path)
        lso_io_utils.save_type_name_to_path(obj=data, path=path)
        x_file_path = os.path.join(path, X_FILE_NAME)
        np.save(x_file_path, data.x)
        if data.objective is not None:
            objective_file_path = os.path.join(path, OBJECTIVE_FILE_NAME)
            np.save(objective_file_path, data.objective)
        if data.features is not None:
            features_file_path = os.path.join(path, FEATURES_FILE_NAME)
            np.save(features_file_path, data.features)
        return
    raise ValueError(f'Unknown serialization scheme for {data}.')


def load_data(path: str):
    # You might want to insert a code for a new deserialization before this default behavior for a NumpyData.
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    Warning(f'Default NumpyData deserialization run for {type_name}.')
    x_file_path = os.path.join(path, LOAD_X_FILE_NAME_NP)
    x = np.load(x_file_path)
    objective_file_path = os.path.join(path, LOAD_OBJECTIVE_FILE_NAME_NP)
    features_file_path = os.path.join(path, LOAD_FEATURES_FILE_NAME_NP)
    objective = np.load(objective_file_path) if os.path.exists(objective_file_path) else None
    features = np.load(features_file_path) if os.path.exists(features_file_path) else None
    return DATA_NAME_TO_TYPE[type_name](x=x, objective=objective, features=features)


def save_multiple_datasets(datasets: List[lso_data.Data], path: str):
    lso_io_utils.create_path(path)
    for idx, dataset in enumerate(datasets):
        current_dataset_path = os.path.join(path, str(idx))
        save_data(dataset, current_dataset_path)


def load_multiple_datasets(path: str):
    nb_of_datasets = len(os.listdir(path))
    datasets = []

    for idx in range(nb_of_datasets):
        current_dataset_path = os.path.join(path, str(idx))
        datasets.append(load_data(current_dataset_path))

    return datasets
