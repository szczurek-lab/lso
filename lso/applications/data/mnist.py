from lso.data import data as lso_data
from lso.data import numpy_data as lso_np_data
from lso.data import vectorizer as lso_data_vectorizer


class MNISTNumpyData(lso_np_data.NumpyData):
    pass


class MNISTFlattenVectorizer(lso_data_vectorizer.Vectorizer):

    def encode(self, data: lso_data.Data) -> lso_np_data.NumpyData:
        if not isinstance(data, MNISTNumpyData):
            raise ValueError(f'MNISTFlattenVectorizer encode called for not MNISTNumpyData but for {data}.')
        return lso_np_data.NumpyData(
            x=data.x.reshape((data.x.shape[0], -1)),
            features=data.features,
            objective=data.objective,
        )

    def decode(self, data: lso_np_data.NumpyData) -> MNISTNumpyData:
        return MNISTNumpyData(
            x=data.x.reshape((-1, 28, 28)),
            objective=data.objective,
            features=data.features
        )

    def get_config_dict(self):
        return {}
