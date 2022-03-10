# Model, its instance. Instance and its params.

As we want to model a model trajectory we need to move away from a classic track of NN model 
development. This is because - in order to store many models - we cannot keep them all together in memory. Therefore, we introduce the following ontology of NN models.

    Model + InstanceParams -> ModelInstance
    ModelInstance + Data/Latent -> Latent/Data 

## `Model`
One may think about the `Model` as simply architecture of the model (e.g., layers shapes,
control flow, etc.). In order to have something that we can run computations on (e.g., compute latent for data)
we need to `get_instance` of the `Model`. We expect two ways of obtaining instances: one for a totally new instance (
in this case we do not pass any params to `get_instance`) and other for which we provide previously obtained `instance_params`
(think e.g., about model deserialization when one loads params from memory and passes them to a `Model`).

## `ModelInstance`

`ModelInstance` is a model architecture extended with everything needed to perform computations on `Data/Latent`. It can
accept `Data` and `encode`s it into `Latent` and the opposite is done by a `decode` method.


