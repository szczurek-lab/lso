from abc import ABC
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Tuple


@dataclass
class Data(ABC):
    x: Any
    objective: Optional[Any] = None
    features: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple:
        x_idx = self.x[idx]
        objective_idx = self.objective[idx] if self.objective is not None else None
        features_idx = self.features[idx] if self.features is not None else None
        return x_idx, objective_idx, features_idx

    def __add__(self, other: Any) -> "Data":
        raise NotImplementedError


@dataclass
class Latent:
    z: Any
    objective: Optional[Any] = None
    features: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int) -> Tuple:
        x_idx = self.z[idx]
        objective_idx = self.objective[idx] if self.objective is not None else None
        features_idx = self.features[idx] if self.features is not None else None
        return x_idx, objective_idx, features_idx

    def __add__(self, other: Any) -> "Latent":
        raise NotImplementedError
