import numpy as np
from typing import Optional

from lso.applications.utils.squares import utils
from lso.data import data as lso_data
from lso.objective_function import objective_function as lso_objective_function


class PenalizedJaccardShapeArea(lso_objective_function.ObjectiveFunction):

    def __init__(self, shape_likeliness_coeff: Optional[float] = 1.0):
        self.shape_likeliness_coeff = shape_likeliness_coeff

    def calculate_jaccard_penalty(self, x: np.array) -> np.array:

        # Compute the Jaccard distance for areas of: shape x and the square circumscribed on shape x.

        area_x = utils.calculate_area(x)
        area_circumscribed_x = utils.calculate_circumscribed_square_area(x)

        jaccard_index = area_x / area_circumscribed_x
        jaccard_distance = 1 - jaccard_index

        return jaccard_distance

    def evaluate(self, data: lso_data.Data) -> lso_data.Data:

        assert len(data.x.shape) == 3

        area = utils.calculate_area(data.x)
        penalty = self.shape_likeliness_coeff * self.calculate_jaccard_penalty(data.x)
        objective = area - penalty

        data.objective = objective

        return data
