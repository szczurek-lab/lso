import numpy as np
from dataclasses import dataclass
from typing import Any
from typing import Optional

from lso.data import data as lso_data


@dataclass
class NumpyData(lso_data.Data):
    x: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyData":

        if not issubclass(type(other.x), type(self.x)):
            return NotImplemented

        if not issubclass(type(other.x), type(self.objective)):
            return NotImplemented

        if not issubclass(type(other.features), type(self.features)):
            return NotImplemented

        return type(self)(
            x=np.concatenate((self.x, other.x)),
            objective=np.concatenate((self.objective, other.objective)),
            features=np.concatenate((self.features, other.features))
        )


@dataclass
class NumpyLatent(lso_data.Latent):
    z: np.array
    objective: Optional[np.array] = None
    features: Optional[np.array] = None

    def __add__(self, other: Any) -> "NumpyLatent":

        if not issubclass(type(other.z), type(self.z)):
            return NotImplemented

        if not issubclass(type(other.x), type(self.objective)):
            return NotImplemented

        if not issubclass(type(other.features), type(self.features)):
            return NotImplemented

        return type(self)(
            z=np.concatenate((self.z, other.z)),
            objective=np.concatenate((self.objective, other.objective)),
            features=np.concatenate((self.features, other.features))
        )
