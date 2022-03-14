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
        if not isinstance(self.x, type(other.x)):
            raise ValueError(f'cannot add {type(self.x)} to {type(other.x)}')

        if not isinstance(self.objective, type(other.objective)):
            raise ValueError(f'cannot add {type(self.objective)} to {type(other.objective)}')

        if not isinstance(self.features, type(other.features)):
            raise ValueError(f'cannot add {type(self.features)} to {type(other.features)}')

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
        if not isinstance(self.z, type(other.z)):
            raise ValueError(f'cannot add {type(self.x)} to {type(other.x)}')

        if not isinstance(self.objective, type(other.objective)):
            raise ValueError(f'cannot add {type(self.objective)} to {type(other.objective)}')

        if not isinstance(self.features, type(other.features)):
            raise ValueError(f'cannot add {type(self.features)} to {type(other.features)}')

        return type(self)(
            z=np.concatenate((self.z, other.z)),
            objective=np.concatenate((self.objective, other.objective)),
            features=np.concatenate((self.features, other.features))
        )
