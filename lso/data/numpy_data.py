import numpy as np
from dataclasses import dataclass
from typing import Any
from typing import Optional

from lso.data import data as lso_data


# noinspection PyDataclass
@dataclass
class NumpyData(lso_data.Data):
    x: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyData":

        if not issubclass(type(other.x), type(self.x)):
            return NotImplemented

        if not issubclass(type(other.objective), type(self.objective)):
            return NotImplemented

        if not issubclass(type(other.features), type(self.features)):
            return NotImplemented

        x = np.concatenate((self.x, other.x))

        if self.objective is None:
            objective = None
        else:
            objective = np.concatenate((self.objective, other.objective))

        if self.features is None:
            features = None
        else:
            features = np.concatenate((self.features, other.features))

        return type(self)(x=x, objective=objective, features=features)

# noinspection PyDataclass
@dataclass
class NumpyLatent(lso_data.Latent):
    z: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyLatent":

        if not issubclass(type(other.z), type(self.z)):
            return NotImplemented

        if not issubclass(type(other.objective), type(self.objective)):
            return NotImplemented

        if not issubclass(type(other.features), type(self.features)):
            return NotImplemented

        z = np.concatenate((self.z, other.z))

        if self.objective is None:
            objective = None
        else:
            objective = np.concatenate((self.objective, other.objective))

        if self.features is None:
            features = None
        else:
            features = np.concatenate((self.features, other.features))

        return type(self)(z=z, objective=objective, features=features)
    