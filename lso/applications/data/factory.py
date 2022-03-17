from torchvision.datasets import mnist
import numpy as np

from lso.applications.data import mnist as app_mnist
from lso.applications.objectives import mnist as lso_app_obj_mnist


class DataFactory:

    @classmethod
    def get_from_args(cls, args):

        if args['data'] == 'MNIST_TRAIN_AREA':
            return cls.get_initial_mnist_data_with_area_objective()

    @classmethod
    def get_initial_mnist_data_with_area_objective(cls):

        mnist_data = mnist.MNIST(root='./data', train=True, download=True, transform=None)

        x = mnist_data.test_data.numpy() / 255.0
        labels = mnist_data.train_labels.numpy()

        raw_data = app_mnist.MNISTNumpyData(
            x=x.astype(np.float32),
            features=labels.astype(np.float32),
        )

        return lso_app_obj_mnist.MNISTSumObjectiveFunction().evaluate(raw_data)


class VectorizerFactory:

    @classmethod
    def get_from_args(cls, args):
        if args['name'] == 'MNISTFlattenVectorizer':
            return app_mnist.MNISTFlattenVectorizer()
