# Objective function

In this directory we will provide different `ObjectiveFunction` implementations. A base interface
of an `ObjectiveFunction` aims at enriching `Data` with objective values.

##`ObjectiveFunction`
- `evaluate(self, data: Data) -> Data`: evaluate a `Data` and provide a `Data` with the same examples but enriched 
with `objective` values.
