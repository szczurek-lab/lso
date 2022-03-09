# DATA

In this directory all objects necessary to model data points and data management are implemented.

## `Data` class
The base object is `Data` that models a single dataset with entities we would like to explore.
It is a `dataclass` with a following attributes:
- `x` that is a collection of data points (e.g., list, `np.array`, etc.),
- [Optional] `objective` that is a collection of the same length as `x` and models the values of an optimized objective.
- [Optional] `features` that is a collection of the same length as `x` and models additional features about the data.

It has the following methods:
- `__len__(self)`: that outputs number of data points,
- `__add__(self, other: Data)`: that given a new `other` chunk of appropriate `Data` creates a new dataset that consist of old and newly added points,
- `__getitem__(self, idx: int)`: that returns a tuple `(x, label, features)` for an i-th element in a dataset.

## `Latent` class
The base object is `Latent` that models an encoding of `Data` objects to the latent space.
It is a `dataclass` with a following attributes:
- `z` that is a collection of embedded data points (e.g., list, `np.array`, etc.),
- [Optional] `objective` that is a collection of the same length as `z` and models the values of an optimized objective.
- [Optional] `features` that is a collection of the same length as `z` and models additional features about the data.

It has the following methods:
- `__len__(self)`: that outputs number of data points,
- `__add__(self, other: Latent)`: that given a new `other` chunk of appropriate `Latent` creates a new dataset that consist of old and newly added points,
- `__getitem__(self, idx: int)`: that returns a tuple `(z, label, features)` for an i-th element in a dataset.
