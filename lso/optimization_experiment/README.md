# Optimization Experiment and its Result 

In this directory we will provide implementation for `OptimizationExperiment` and its result `OptimizationExperimentResult`.
Optimization Experiment orchestrates the cooperation of all managers: `DataManager` that stores the data trajectory and initial data,
`ModelManager` that stores the model trajectory, `OptimizerManager` that manages the optimizer trajectory and finally `EvaluationFunction`
that evaluates candidates provided by the `Optimizer`. After running the experiment - its `OptimizationExperimentResuls` 
stores all the managers.

## `OptimizationExperiment`
- `run(self, data_manager: lso_data_manager.DataManager, model_manager: lso_model_manager.ModelManager, optimizer: lso_optimizer_manager.OptimizerManager, objective_function: lso_objective_function.ObjectiveFunction) -> OptimizationExperimentResult`: run the experiment using data provided by `DataManager`, models provided by `ModelManager`. Get candidates using `OptimizerManager` and evalute them using `ObjectiveFunction`.
