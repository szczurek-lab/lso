import numpy as np

from lso.optimizer_manager import optimizer_manager as lso_om
from lso.data import data as lso_data
from lso.data import numpy_data as lso_np_data
from lso.data_manager import data_manager as lso_dm
from lso.model_manager import model_manager as lso_mm


class SampleAroundTopKOptimizerManager(lso_om.OptimizerManager):

    def __init__(self, sigma: float, k: int):
        self.sigma = sigma
        self.k = k

    def get_candidates(
            self,
            model_manager: lso_mm.ModelManager,
            data_manager: lso_dm.DataManager,
            epoch_nb: int,
            nb_of_candidates: int
    ) -> lso_data.Data:
        all_data = data_manager.get_all_data()
        top_data_points = all_data.x[all_data.objective.argsort()[-self.k:]]
        encoded = model_manager.encode(type(all_data)(x=top_data_points), epoch_nb)

        noise = np.random.normal(loc=0, scale=self.sigma, size=encoded.z.shape).astype('float32')
        z = encoded.z + noise

        candidates_data = lso_np_data.NumpyLatent(z=z)
        data = model_manager.decode(candidates_data, epoch_nb)

        return data

    def get_config_dict(self):
        return {'sigma': self.sigma, 'k': self.k}
