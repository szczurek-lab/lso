# Model, its instance. Instance and its params.

As we want to model a model trajectory we need to move away from a classic track of NN model 
development. This is because - in order to store many models - we cannot keep them all together in memory. Therefore, we introduce the following ontology of NN models.

    Model + InstanceParams -> ModelInstance
    ModelInstance + Data/Latent -> Latent/Data 

## `Model`
One may think about the `Model` as the architecture of the model (e.g., layers, shapes,
control flow, etc.). In order to have something that we can run computations on (e.g., compute the latent space for the data)
we need to `get_instance` of the `Model`. We expect two ways of obtaining instances: one for a novel instance (
in this case we do not pass any params to `get_instance`) and other for which we provide previously obtained `instance_params`
(think e.g., about model deserialization when one loads params from memory and passes them to a `Model`).

### Note on serialization

Each `Model` serializes to `JSON` with `get_config_dict` method and deserializes
from JSON using `from_config_dict` `classmethod`.

## `ModelInstance`

`ModelInstance` is a model architecture extended with everything needed to perform computations on `Data/Latent`. It
accept `Data` and `encode`s it into `Latent` and the opposite is done by a `decode` method.


