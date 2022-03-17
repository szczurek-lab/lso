from lso.applications.data import factory as lso_data_factory
from lso.data_manager import data_manager as lso_data_manager


class DataManagerFactory:

    @classmethod
    def get_from_args(cls, args):
        if args['name'] == 'MNIST_TRAIN_AREA':
            return lso_data_manager.BaseDataManager(
                initial_data=lso_data_factory.DataFactory.get_initial_mnist_data_with_area_objective())
