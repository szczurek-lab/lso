import numpy as np
from dataclasses import dataclass
from typing import Any
from typing import Optional

from torch.utils import data as torch_data

from lso.data import data as lso_data


@dataclass
class NumpyData(lso_data.Data, torch_data.Dataset):
    x: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyData":

        if not isinstance(other, NumpyData):
            return NotImplemented

        if not self.x.shape[1:] == other.x.shape[1:]:
            raise ValueError(f'x shapes of added Data: {self}, {other}'
                             f' do not match: {self.x.shape}, {other.x.shape}.')
        x = np.concatenate((self.x, other.x))

        if self.objective is None or other.objective is None:
            objective = None
        else:
            if not self.objective.shape[1:] == other.objective.shape[1:]:
                raise ValueError(f'objective shapes of added Data: {self}, {other}'
                                 f' do not match: {self.objective.shape}, {other.objective.shape}.')
            objective = np.concatenate((self.objective, other.objective))

        if self.features is None or other.features is None:
            features = None
        else:
            if not self.features.shape[1:] == other.features.shape[1:]:
                raise ValueError(f'features shapes of added Data: {self}, {other}'
                                 f' do not match: {self.features.shape}, {other.features.shape}.')
            features = np.concatenate((self.features, other.features))

        return type(self)(x=x, objective=objective, features=features)


@dataclass
class NumpyLatent(lso_data.Latent, torch_data.Dataset):
    z: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyLatent":

        if not isinstance(other, NumpyLatent):
            return NotImplemented

        if not self.z.shape[1:] == other.z.shape[1:]:
            raise ValueError(f'Z shapes of added latents: {self}, {other}'
                             f' do not match: {self.z.shape}, {other.z.shape}.')
        z = np.concatenate((self.z, other.z))

        if self.objective is None or other.objective is None:
            objective = None
        else:
            if not self.objective.shape[1:] == other.objective.shape[1:]:
                raise ValueError(f'objective shapes of added latents: {self}, {other}'
                                 f' do not match: {self.objective.shape}, {other.objective.shape}.')
            objective = np.concatenate((self.objective, other.objective))

        if self.features is None or other.features is None:
            features = None
        else:
            if not self.features.shape[1:] == other.features.shape[1:]:
                raise ValueError(f'features shapes of added latents: {self}, {other}'
                                 f' do not match: {self.features.shape}, {other.features.shape}.')
            features = np.concatenate((self.features, other.features))

        return type(self)(z=z, objective=objective, features=features)
    