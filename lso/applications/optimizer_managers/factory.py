from typing import Dict

from lso.applications.optimizer_managers import sample_around_top_k as lso_app_om_satk
from lso.optimizer_manager import optimizer_manager as lso_om


class OptimizerManagerFactory:

    @classmethod
    def get_from_args(cls, args: Dict) -> lso_om.OptimizerManager:
        if args['name'] == 'SampleAroundTopKOptimizerManager':
            return lso_app_om_satk.SampleAroundTopKOptimizerManager(
                k=args['k'],
                sigma=args['sigma'],
            )
        raise ValueError(f'Unknown args for OptimizerManagerFactory {args}.')
