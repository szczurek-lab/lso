from functools import singledispatch

from lso.applications.data import mnist
from lso.data import vectorizer as lso_vectorizer
from lso.utils import io as lso_io_utils

# Insert new vectorizer types here:
VECTORIZER_TYPES = [
    mnist.MNISTFlattenVectorizer,
]
VECTORIZER_NAME_TO_TYPE = {type_.__name__: type_ for type_ in VECTORIZER_TYPES}


@singledispatch
def save_vectorizer(vectorizer: lso_vectorizer.Vectorizer, path: str):
    Warning(f'Default model serialization run for {type(vectorizer).__name__}.')
    lso_io_utils.create_path(path=path)
    lso_io_utils.save_type_name_to_path(obj=vectorizer, path=path)
    lso_io_utils.save_config_dict_to_path(config_dict=vectorizer.get_config_dict(), path=path)


def load_vectorizer(path: str):
    type_name = lso_io_utils.load_type_name_from_path(path=path)
    config_dict = lso_io_utils.load_config_dict_from_path(path=path)
    return VECTORIZER_NAME_TO_TYPE[type_name](**config_dict)
