from lso.applications.data import mnist as lso_app_mnist
from lso.data import data as lso_data
from lso.objective_function import objective_function as lso_objective_function


class MNISTSumObjectiveFunction(lso_objective_function.ObjectiveFunction):

    def evaluate(self, data: lso_data.Data) -> lso_app_mnist.MNISTNumpyData:
        if not isinstance(data, lso_app_mnist.MNISTNumpyData):
            raise ValueError(f'MNISTSumObjectiveFunction accepts MNISTNumpyData, but {data} provided.')
        print(data.x.shape)
        objective = data.x.sum(axis=(1, 2)).astype('float32')
        data.objective = objective
        return data
