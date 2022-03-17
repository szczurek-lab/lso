from lso.applications.data import factory as lso_app_data_factory
from lso.applications.models.torch import factory as lso_app_t_factory
from lso.applications.model_managers.torch import model_manager as lso_t_model_manager


class ModelManagerFactory:

    @classmethod
    def get_from_args(cls, args):

        if args['name'] == 'BASIC_SIMPLE_PL_PCA_MNIST':
            dflt_nb_of_ep_step = 50
            dflt_initial = 10
            return lso_t_model_manager.BasicPLSingleModelModelManager(
                model=lso_app_t_factory.ModelFactory.get_from_args(args=args['model']),
                batch_size=args['batch_size'],
                nb_of_steps_per_epoch=args['nb_of_steps_per_epoch'] if 'nb_of_steps_per_epoch' in args else dflt_nb_of_ep_step,
                nb_of_initial_epochs=args['nb_of_initial_epochs'] if 'nb_of_initial_epochs' in args else dflt_initial,
                vectorizer=lso_app_data_factory.VectorizerFactory.get_from_args(args['vectorizer']),
            )
