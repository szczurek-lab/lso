from functools import singledispatch
import os

from lso.applications.data import data_serialization
from lso.data_manager import data_manager as lso_dm
import lso.utils.io as lso_io_utils

INITIAL_DATA_SUBPATH = 'initial_data'
EPOCH_DATA_SUBPATH = 'epoch_data'


@singledispatch
def save_data_manager(data_manager: lso_dm.DataManager, path: str):
    raise ValueError(f'ModelManager serialization not defined for {data_manager}.')


@save_data_manager.register(lso_dm.BaseDataManager)
def save_base_data_manager(data_manager: lso_dm.BaseDataManager, path: str):
    lso_io_utils.create_path(path=path)
    lso_io_utils.save_type_name_to_path(obj=data_manager, path=path)

    initial_data_path = os.path.join(path, INITIAL_DATA_SUBPATH)
    epoch_data_path = os.path.join(path, EPOCH_DATA_SUBPATH)

    lso_io_utils.create_path(initial_data_path)
    data_serialization.save_data(data_manager.initial_data, path=initial_data_path)

    lso_io_utils.create_path(epoch_data_path)
    data_serialization.save_multiple_datasets(data_manager.datas, path=epoch_data_path)


def load_data_manager(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    if type_name == 'BaseDataManager':
        return load_base_data_manager(path=path)
    raise ValueError(f'Unknown deserialization procedure for a {type_name}.')


def load_base_data_manager(path: str):
    initial_data_path = os.path.join(path, INITIAL_DATA_SUBPATH)
    epoch_data_path = os.path.join(path, EPOCH_DATA_SUBPATH)

    initial_data = data_serialization.load_data(initial_data_path)
    epoch_data = data_serialization.load_multiple_datasets(epoch_data_path)

    return lso_dm.BaseDataManager(
        initial_data=initial_data,
        datas=epoch_data
    )
