from typing import Dict

from lso.applications.models.torch import pca_models as lso_t_pca_models
from lso.model import model as lso_model


MNIST_INPUT_SIZE = 28 * 28


class ModelFactory:

    @classmethod
    def get_from_args(cls, args: Dict) -> lso_model.Model:
        if args['name'] == 'SIMPLE_PL_PCA_MNIST':
            return lso_t_pca_models.SimplePCATorchModel(latent_dim=args['latent_dim'], input_dim=MNIST_INPUT_SIZE)
        raise ValueError(f'Unknown model name {args["name"]} for a ModelFactory.')
