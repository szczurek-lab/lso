import torch
from typing import Any
from typing import Optional

from lso.data import data as lso_data


class TorchData(lso_data.Data):
    x: torch.Tensor
    objective: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

    def __add__(self, other: Any) -> "TorchData":

        if not isinstance(self.x, type(other.x)):
            raise ValueError(f'cannot add {type(self.x)} to {type(other.x)}')

        if not isinstance(self.objective, type(other.objective)):
            raise ValueError(f'cannot add {type(self.objective)} to {type(other.objective)}')

        if not isinstance(self.features, type(other.features)):
            raise ValueError(f'cannot add {type(self.features)} to {type(other.features)}')

        return type(self)(
            x=torch.cat((self.x, other.x)),
            objective=torch.cat((self.objective, other.objective)),
            features=torch.cat((self.features, other.features))
        )
