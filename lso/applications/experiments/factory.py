from typing import Dict

from lso.applications.experiments import base_experiments as lso_be


class OptimizationExperimentFactory:

    @classmethod
    def get_from_args(cls, args: Dict):
        if args['name'] == 'BaseOptimizationExperiment':
            return lso_be.BaseOptimizationExperiment(
                nb_of_epochs=args['nb_of_epochs'],
                nb_of_candidates_per_epoch=args['nb_of_candidates_per_epoch'],
            )
