from typing import Dict

import torch


class PytorchInstanceParams:

    def __init__(self, state_dict: Dict):
        self.state_dict = state_dict
