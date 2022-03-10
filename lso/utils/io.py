import json
import os
from typing import Any


CONFIG_DICT_FILE_NAME = 'config_dict.json'
TYPE_FILE_NAME = 'type.txt'


def create_path(path: str):
    os.makedirs(path, exist_ok=True)


def save_type_name_to_path(obj: Any, path: str, type_file_name: str = TYPE_FILE_NAME):
    full_type_file_path = os.path.join(path, type_file_name)
    with open(full_type_file_path, 'w') as file_handle:
        file_handle.write(str(type(obj).__name__))


def load_type_name_from_path(path: str, type_file_name: str = TYPE_FILE_NAME):
    full_type_file_path = os.path.join(path, type_file_name)
    with open(full_type_file_path, 'r') as file_handle:
        return file_handle.read()


def save_config_dict_to_path(config_dict, path: str, config_dict_file_name: str = CONFIG_DICT_FILE_NAME):
    full_config_dict_path = os.path.join(path, config_dict_file_name)
    with open(full_config_dict_path, 'w') as out_file:
        json.dump(config_dict, out_file)


def load_config_dict_from_path(path: str, config_dict_file_name: str = CONFIG_DICT_FILE_NAME):
    full_config_dict_path = os.path.join(path, config_dict_file_name)
    with open(full_config_dict_path, 'r') as out_file:
        return json.load(out_file)
